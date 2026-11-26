import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, field_validator

from app.agents.chat_agent import create_chat_model
from app.core.config import settings
from app.memory.episodic import EpisodicMemoryService
from app.memory.semantic import SemanticMemoryService

SENSITIVE_PATTERNS = (
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-[0-9A-Za-z\-_]{16,}\b"),
    re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[0-9A-Za-z\-._~+/]+=*\b", re.IGNORECASE),
    re.compile(r"\bpostgres(?:ql)?(?:\+\w+)?://[^/\s:@]+:[^/\s@]+@", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
)

SYS_PROMPT_MEMORY_MANAGER = """
You are a memory management assistant for a conversational AI system.

Analyze the latest user and assistant turn and decide what (if anything) should be stored, updated, or deactivated in long-term memory.

Memory classification rules:
1. Semantic memory (`semantic_actions`):
   - Store durable, reusable information explicitly stated by the user:
     * Stable user facts (name, role/occupation, location, background)
     * User preferences (communication style, preferred tools, likes/dislikes)
     * Technical skills (languages, frameworks, databases, tools)
     * Long-term goals and active projects
     * Important architectural or workflow decisions
   - Categories allowed: "fact", "preference", "skill", "goal", "project", "decision".
   - If an existing semantic memory is provided and the new turn refines, corrects, or supersedes it, use action="update" with its `target_memory_id` (or action="deactivate" if the user explicitly states it is no longer true).
   - Set `is_factual=true` ONLY for explicitly stated facts, never assumptions, guesses, or AI-generated speculation.

2. Episodic memory (`episodic_actions`):
   - Store significant interaction summaries, milestones, decisions made in this session, or notable experiences worth recalling chronologically.
   - Do NOT store trivial greetings ("hi", "hello", "thanks"), small talk, or temporary conversational filler.

3. Security & privacy rules (STRICT):
   - NEVER store passwords, API keys, tokens, database connection strings with credentials, private keys, or sensitive personal data (SSNs, payment card numbers, etc.).
   - If the turn contains only small talk, temporary details, speculation, or sensitive credentials, return `decision="none"` with empty action lists.
""".strip()


def contains_sensitive_data(text: str) -> bool:
    """Return True if text appears to contain secrets, credentials, or sensitive PII."""
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _validate_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


class SemanticMemoryAction(BaseModel):
    """Structured memory mutation proposed by the memory management LLM."""

    action: Literal["create", "update", "deactivate"]
    content: str | None = None
    category: Literal["fact", "preference", "skill", "goal", "project", "decision"] | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    target_memory_id: str | None = None
    is_factual: bool = True

    @field_validator("content")
    @classmethod
    def _clean_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class EpisodicMemoryAction(BaseModel):
    """Structured episodic event summary proposed by the memory management LLM."""

    summary: str
    event_type: Literal[
        "conversation_summary",
        "decision",
        "milestone",
        "experience",
        "other",
    ] = "experience"
    is_significant: bool = True

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        return _validate_non_empty_text(value, "Summary")


class MemoryManagementPlan(BaseModel):
    """Structured output returned by the memory classification model."""

    decision: Literal["semantic", "episodic", "both", "none"]
    semantic_actions: list[SemanticMemoryAction] = Field(default_factory=list)
    episodic_actions: list[EpisodicMemoryAction] = Field(default_factory=list)
    reasoning: str = ""


class UpsertSemanticToolInput(BaseModel):
    content: str = Field(..., description="Factual semantic statement to store or deduplicate.")
    category: Literal["fact", "preference", "skill", "goal", "project", "decision"] | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class UpdateSemanticToolInput(BaseModel):
    memory_id: str = Field(..., description="UUID string of the existing semantic memory.")
    content: str | None = None
    category: Literal["fact", "preference", "skill", "goal", "project", "decision"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DeactivateSemanticToolInput(BaseModel):
    memory_id: str = Field(..., description="UUID string of the semantic memory to deactivate.")


class CreateEpisodicToolInput(BaseModel):
    summary: str = Field(..., description="Concise summary of the significant experience or event.")
    event_type: Literal[
        "conversation_summary",
        "decision",
        "milestone",
        "experience",
        "other",
    ] = "experience"


@dataclass(slots=True)
class MemoryExecutionReport:
    """Summary of memory-management actions executed for a conversation turn."""

    decision: str
    results: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_memory_tools(
    user_id: str,
    session_id: str,
    semantic_service: SemanticMemoryService,
    episodic_service: EpisodicMemoryService,
) -> dict[str, BaseTool]:
    """
    Build user- and session-scoped LangChain tools wrapping memory services.

    `user_id` and `session_id` are validated and bound via closure so the model
    can never supply or override tenant identity.
    """
    validated_user_id = _validate_non_empty_text(user_id, "User ID")
    validated_session_id = _validate_non_empty_text(session_id, "Session ID")

    def upsert_semantic_memory(
        content: str,
        category: str | None = None,
        confidence: float = 0.85,
    ) -> str:
        try:
            cleaned_content = _validate_non_empty_text(content, "Content")
            if contains_sensitive_data(cleaned_content):
                return json.dumps(
                    {"status": "rejected", "reason": "sensitive_data_detected"}
                )
            memory, status = semantic_service.upsert_memory(
                user_id=validated_user_id,
                content=cleaned_content,
                category=category,
                confidence=confidence,
                source_session_id=validated_session_id,
            )
            return json.dumps(
                {
                    "status": status,
                    "memory_id": str(memory.id),
                    "type": "semantic",
                }
            )
        except Exception as error:
            return json.dumps({"status": "error", "error": str(error)})

    def update_semantic_memory(
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
    ) -> str:
        try:
            parsed_id = UUID(str(memory_id))
            if content is not None and contains_sensitive_data(content):
                return json.dumps(
                    {"status": "rejected", "reason": "sensitive_data_detected"}
                )
            updated = semantic_service.update_memory(
                memory_id=parsed_id,
                user_id=validated_user_id,
                content=content,
                category=category,
                confidence=confidence,
            )
            if updated is None:
                return json.dumps({"status": "not_found", "memory_id": str(parsed_id)})
            return json.dumps(
                {
                    "status": "updated",
                    "memory_id": str(updated.id),
                    "type": "semantic",
                }
            )
        except Exception as error:
            return json.dumps({"status": "error", "error": str(error)})

    def deactivate_semantic_memory(memory_id: str) -> str:
        try:
            parsed_id = UUID(str(memory_id))
            deactivated = semantic_service.deactivate_memory(
                memory_id=parsed_id,
                user_id=validated_user_id,
            )
            if deactivated is None:
                return json.dumps({"status": "not_found", "memory_id": str(parsed_id)})
            return json.dumps(
                {
                    "status": "deactivated",
                    "memory_id": str(deactivated.id),
                    "type": "semantic",
                }
            )
        except Exception as error:
            return json.dumps({"status": "error", "error": str(error)})

    def create_episodic_memory(
        summary: str,
        event_type: str = "experience",
    ) -> str:
        try:
            cleaned_summary = _validate_non_empty_text(summary, "Summary")
            if contains_sensitive_data(cleaned_summary):
                return json.dumps(
                    {"status": "rejected", "reason": "sensitive_data_detected"}
                )
            memory, created = episodic_service.create_memory_if_new(
                user_id=validated_user_id,
                summary=cleaned_summary,
                session_id=validated_session_id,
                event_type=event_type,
            )
            return json.dumps(
                {
                    "status": "created" if created else "unchanged",
                    "memory_id": str(memory.id),
                    "type": "episodic",
                }
            )
        except Exception as error:
            return json.dumps({"status": "error", "error": str(error)})

    return {
        "upsert_semantic_memory": StructuredTool.from_function(
            func=upsert_semantic_memory,
            name="upsert_semantic_memory",
            description="Create or deduplicate/update a factual semantic memory for the current user.",
            args_schema=UpsertSemanticToolInput,
        ),
        "update_semantic_memory": StructuredTool.from_function(
            func=update_semantic_memory,
            name="update_semantic_memory",
            description="Update an existing semantic memory owned by the current user.",
            args_schema=UpdateSemanticToolInput,
        ),
        "deactivate_semantic_memory": StructuredTool.from_function(
            func=deactivate_semantic_memory,
            name="deactivate_semantic_memory",
            description="Soft-deactivate an outdated or invalidated semantic memory owned by the current user.",
            args_schema=DeactivateSemanticToolInput,
        ),
        "create_episodic_memory": StructuredTool.from_function(
            func=create_episodic_memory,
            name="create_episodic_memory",
            description="Record a significant episodic event or milestone for the current user and session.",
            args_schema=CreateEpisodicToolInput,
        ),
    }


class MemoryManagerAgent:
    """
    LLM-powered memory classifier and tool orchestrator.

    Separates structured classification from tool invocation so memory-write
    operations remain independently testable and strictly user-scoped.
    """

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = create_chat_model(settings.chat_model)
        return self._model

    @staticmethod
    def _format_existing_memories(existing_memories: list[Any] | None) -> str:
        if not existing_memories:
            return "None"
        lines = []
        for item in existing_memories:
            memory = item[0] if isinstance(item, tuple) else item
            memory_id = getattr(memory, "id", None)
            content = getattr(memory, "content", None)
            category = getattr(memory, "category", None)
            if content:
                lines.append(
                    f"- id={memory_id} | category={category} | content={content}"
                )
        return "\n".join(lines) if lines else "None"

    def analyze_turn(
        self,
        user_message: str,
        assistant_message: str,
        existing_memories: list[Any] | None = None,
    ) -> MemoryManagementPlan:
        """
        Classify the turn and produce a validated `MemoryManagementPlan`.
        """
        _validate_non_empty_text(user_message, "User message")

        existing_block = self._format_existing_memories(existing_memories)
        prompt = (
            f"Existing relevant semantic memories:\n{existing_block}\n\n"
            f"Latest turn:\n"
            f"USER: {user_message}\n"
            f"ASSISTANT: {assistant_message}"
        )
        structured_model = self._get_model().with_structured_output(
            MemoryManagementPlan
        )
        plan = structured_model.invoke(
            [
                SystemMessage(content=SYS_PROMPT_MEMORY_MANAGER),
                HumanMessage(content=prompt),
            ]
        )
        if isinstance(plan, MemoryManagementPlan):
            return plan
        return MemoryManagementPlan.model_validate(plan)

    def apply_plan(
        self,
        plan: MemoryManagementPlan,
        user_id: str,
        session_id: str,
        semantic_service: SemanticMemoryService,
        episodic_service: EpisodicMemoryService,
    ) -> MemoryExecutionReport:
        """
        Execute a validated `MemoryManagementPlan` via user-scoped LangChain tools.
        """
        tools = build_memory_tools(
            user_id=user_id,
            session_id=session_id,
            semantic_service=semantic_service,
            episodic_service=episodic_service,
        )
        report = MemoryExecutionReport(decision=plan.decision)

        if plan.decision == "none":
            return report

        if plan.decision in {"semantic", "both"}:
            for item in plan.semantic_actions:
                if not item.is_factual:
                    report.skipped.append("non_factual_semantic_candidate")
                    continue
                if item.content and contains_sensitive_data(item.content):
                    report.skipped.append("sensitive_semantic_candidate")
                    continue

                if item.action == "create":
                    if not item.content:
                        report.skipped.append("empty_semantic_create")
                        continue
                    raw_result = tools["upsert_semantic_memory"].invoke(
                        {
                            "content": item.content,
                            "category": item.category,
                            "confidence": item.confidence,
                        }
                    )
                elif item.action == "update":
                    if not item.target_memory_id:
                        if item.content:
                            raw_result = tools["upsert_semantic_memory"].invoke(
                                {
                                    "content": item.content,
                                    "category": item.category,
                                    "confidence": item.confidence,
                                }
                            )
                        else:
                            report.skipped.append("missing_target_memory_id")
                            continue
                    else:
                        raw_result = tools["update_semantic_memory"].invoke(
                            {
                                "memory_id": item.target_memory_id,
                                "content": item.content,
                                "category": item.category,
                                "confidence": item.confidence,
                            }
                        )
                else:  # deactivate
                    if not item.target_memory_id:
                        report.skipped.append("missing_target_memory_id")
                        continue
                    raw_result = tools["deactivate_semantic_memory"].invoke(
                        {"memory_id": item.target_memory_id}
                    )

                self._record_tool_outcome(raw_result, report)

        if plan.decision in {"episodic", "both"}:
            for item in plan.episodic_actions:
                if not item.is_significant:
                    report.skipped.append("insignificant_episodic_candidate")
                    continue
                if contains_sensitive_data(item.summary):
                    report.skipped.append("sensitive_episodic_candidate")
                    continue

                raw_result = tools["create_episodic_memory"].invoke(
                    {
                        "summary": item.summary,
                        "event_type": item.event_type,
                    }
                )
                self._record_tool_outcome(raw_result, report)

        return report

    @staticmethod
    def _record_tool_outcome(raw_result: str, report: MemoryExecutionReport) -> None:
        try:
            parsed = json.loads(raw_result)
        except Exception:
            report.errors.append(str(raw_result))
            return

        if parsed.get("status") == "error":
            report.errors.append(parsed.get("error", "Unknown tool error"))
        elif parsed.get("status") == "rejected":
            report.skipped.append(parsed.get("reason", "rejected"))
        else:
            report.results.append(parsed)

    def process_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        semantic_service: SemanticMemoryService,
        episodic_service: EpisodicMemoryService,
        existing_memories: list[Any] | None = None,
    ) -> MemoryExecutionReport:
        """
        Analyze a completed conversation turn and persist validated memories.
        """
        _validate_non_empty_text(user_id, "User ID")
        _validate_non_empty_text(session_id, "Session ID")

        if not user_message or not user_message.strip():
            return MemoryExecutionReport(decision="none")

        if contains_sensitive_data(user_message) and contains_sensitive_data(
            assistant_message
        ):
            return MemoryExecutionReport(
                decision="none",
                skipped=["sensitive_turn_content"],
            )

        plan = self.analyze_turn(
            user_message=user_message,
            assistant_message=assistant_message,
            existing_memories=existing_memories,
        )
        return self.apply_plan(
            plan=plan,
            user_id=user_id,
            session_id=session_id,
            semantic_service=semantic_service,
            episodic_service=episodic_service,
        )
