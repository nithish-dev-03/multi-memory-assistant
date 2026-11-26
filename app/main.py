import argparse
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.agents.chat_agent import create_chat_model
from app.agents.memory_agent import MemoryManagerAgent
from app.core.config import settings
from app.core.database import get_db_session
from app.memory.episodic import EpisodicMemoryService
from app.memory.retrieval import MemoryRetrievalResult, MemoryRetrievalService
from app.memory.semantic import SemanticMemoryService
from app.memory.short_term import ShortTermMemory
from app.utils.chat_services import summarize_history
from app.utils.prompt_budget import PromptBudgetManager


def count_token(model: Any, messages: list[BaseMessage]) -> int:
    """Count tokens in the supplied messages using the active model."""
    return model.get_num_tokens_from_messages(messages)


def resolve_user_id(explicit_user_id: str | None = None) -> str:
    """
    Resolve the active user ID from CLI argument, config, or interactive prompt.
    """
    candidate = explicit_user_id if explicit_user_id is not None else settings.cli_user_id
    if candidate is None:
        candidate = input("User ID: ").strip()

    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("User ID cannot be empty")
    return candidate.strip()


def _retrieve_turn_memories(
    user_id: str,
    query: str,
) -> MemoryRetrievalResult:
    """Retrieve semantic and episodic memories inside a scoped DB session."""
    with get_db_session() as db:
        retrieval_service = MemoryRetrievalService(db)
        return retrieval_service.retrieve_memories(
            user_id=user_id,
            query=query,
            semantic_limit=settings.semantic_retrieval_limit,
            episodic_limit=settings.episodic_retrieval_limit,
            max_distance=settings.semantic_max_distance,
        )


def _persist_turn_memories(
    memory_agent: MemoryManagerAgent,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    existing_memories: list[Any],
) -> None:
    """Persist extracted semantic/episodic memories in an isolated DB session."""
    with get_db_session() as db:
        memory_agent.process_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            semantic_service=SemanticMemoryService(db),
            episodic_service=EpisodicMemoryService(db),
            existing_memories=existing_memories,
        )


def _stream_assistant_response(model: Any, prompt_messages: list[BaseMessage]) -> str:
    """Stream the model response to stdout and return the full response text."""
    print("\nAssistant: ", end="", flush=True)
    ai_response = ""

    for chunk in model.stream(prompt_messages):
        content = chunk.content

        if isinstance(content, str):
            print(content, end="", flush=True)
            ai_response += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text_content = block.get("text", "")
                    if text_content:
                        print(text_content, end="", flush=True)
                        ai_response += text_content

    print("\n")
    return ai_response


def create_convo(
    user_id: str | None = None,
    session_id: str | None = None,
) -> None:
    """Run the interactive multi-memory CLI conversation loop."""
    active_user_id = resolve_user_id(user_id)
    active_session_id = (
        session_id.strip()
        if isinstance(session_id, str) and session_id.strip()
        else settings.cli_session_id
    )

    model = create_chat_model(settings.chat_model)
    memory = ShortTermMemory()
    budget_manager = PromptBudgetManager(
        max_prompt_tokens=settings.max_prompt_tokens,
        response_reserve_tokens=settings.response_reserve_tokens,
        memory_context_max_tokens=settings.memory_context_max_tokens,
    )
    memory_agent = MemoryManagerAgent()

    # Initialize the session with a system message.
    if not memory.get_messages(active_session_id):
        memory.add_message(
            active_session_id,
            SystemMessage(content="You are a helpful assistant."),
        )

    while True:
        text = input("\nYou: ").strip()

        if text.lower() in {"exit", "quit", "bye"}:
            print("Assistant: Goodbye!")
            print("\n--- STORED HISTORY ---")
            for index, message in enumerate(memory.get_messages(active_session_id)):
                print(f"{index}: {message.type} -> {message.content}")
            break

        if not text:
            continue

        # 1. Backup existing short-term history for rollback.
        previous_history = memory.get_messages(active_session_id)
        current_user_message = HumanMessage(content=text)

        try:
            # 2. Retrieve relevant long-term memories (not persisted to short-term memory).
            retrieval_result = _retrieve_turn_memories(
                user_id=active_user_id,
                query=text,
            )

            # 3. Assemble transient prompt within unified token budget.
            budget_result = budget_manager.build_prompt(
                history_messages=previous_history,
                current_user_message=current_user_message,
                semantic_memories=retrieval_result.semantic_memories,
                episodic_memories=retrieval_result.episodic_memories,
                message_token_counter=lambda msgs: count_token(model, msgs),
            )

            # 4. Record the user message and stream the assistant response.
            memory.add_message(active_session_id, current_user_message)
            ai_response = _stream_assistant_response(model, budget_result.messages)

            # 5. Store the assistant response in short-term memory.
            memory.add_message(
                active_session_id,
                AIMessage(content=ai_response),
            )

            # 6. Check short-term history length and summarize if it exceeds the window.
            persisted_history = memory.get_messages(active_session_id)
            tokens = count_token(model, persisted_history)
            print(f"Tokens: {tokens}")

            max_tokens = settings.sliding_window_size
            if tokens > max_tokens:
                persisted_history = summarize_history(
                    persisted_history,
                    max_tokens,
                    current_tokens=tokens,
                )
                memory.replace_messages(active_session_id, persisted_history)

            final_tokens = count_token(model, persisted_history)
            print(f"\nFinal token count: {final_tokens}")
            print(f"Maximum token limit: {max_tokens}")

        except Exception as error:
            print(f"\nError: {error}")
            # Restore the previous short-term conversation state.
            memory.replace_messages(active_session_id, previous_history)
            continue

        # 7. Extract and persist long-term memories safely; failures must not
        # roll back or corrupt the already-completed conversation turn.
        try:
            _persist_turn_memories(
                memory_agent=memory_agent,
                user_id=active_user_id,
                session_id=active_session_id,
                user_message=text,
                assistant_message=ai_response,
                existing_memories=retrieval_result.semantic_memories,
            )
        except Exception as memory_error:
            print(f"\nMemory update warning: {memory_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Memory Assistant CLI")
    parser.add_argument("--user-id", default=None, help="User ID for memory scoping")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID for short-term and episodic memory",
    )
    args = parser.parse_args()
    create_convo(user_id=args.user_id, session_id=args.session_id)


if __name__ == "__main__":
    main()
