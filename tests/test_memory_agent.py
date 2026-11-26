import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.agents.memory_agent import (
    EpisodicMemoryAction,
    MemoryManagementPlan,
    MemoryManagerAgent,
    SemanticMemoryAction,
    build_memory_tools,
    contains_sensitive_data,
)


def test_memory_tools_bind_user_and_session_and_prevent_duplicates():
    semantic_service = MagicMock()
    episodic_service = MagicMock()
    sem_id = uuid4()
    ep_id = uuid4()
    semantic_service.upsert_memory.return_value = (SimpleNamespace(id=sem_id), "created")
    episodic_service.create_memory_if_new.return_value = (SimpleNamespace(id=ep_id), False)

    tools = build_memory_tools(
        user_id="user-1",
        session_id="sess-1",
        semantic_service=semantic_service,
        episodic_service=episodic_service,
    )

    sem_resp = json.loads(
        tools["upsert_semantic_memory"].invoke(
            {"content": "User lives in Chennai", "category": "fact", "confidence": 0.95}
        )
    )
    ep_resp = json.loads(
        tools["create_episodic_memory"].invoke(
            {"summary": "Completed memory design", "event_type": "milestone"}
        )
    )

    assert sem_resp == {"status": "created", "memory_id": str(sem_id), "type": "semantic"}
    assert ep_resp == {"status": "unchanged", "memory_id": str(ep_id), "type": "episodic"}
    semantic_service.upsert_memory.assert_called_once_with(
        user_id="user-1",
        content="User lives in Chennai",
        category="fact",
        confidence=0.95,
        source_session_id="sess-1",
    )
    episodic_service.create_memory_if_new.assert_called_once_with(
        user_id="user-1",
        summary="Completed memory design",
        session_id="sess-1",
        event_type="milestone",
    )


def test_memory_tools_reject_sensitive_data_and_handle_errors_safely():
    semantic_service = MagicMock()
    episodic_service = MagicMock()
    semantic_service.update_memory.side_effect = RuntimeError("db unavailable")

    tools = build_memory_tools(
        user_id="user-1",
        session_id="sess-1",
        semantic_service=semantic_service,
        episodic_service=episodic_service,
    )

    rejected = json.loads(
        tools["upsert_semantic_memory"].invoke(
            {"content": "My key is AIzaSyDummySecretKey1234567890"}
        )
    )
    err = json.loads(
        tools["update_semantic_memory"].invoke(
            {"memory_id": str(uuid4()), "content": "Updated fact"}
        )
    )
    bad_uuid = json.loads(
        tools["deactivate_semantic_memory"].invoke({"memory_id": "not-a-uuid"})
    )

    assert rejected["status"] == "rejected"
    semantic_service.upsert_memory.assert_not_called()
    assert err == {"status": "error", "error": "db unavailable"}
    assert bad_uuid["status"] == "error"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("api_key=supersecretvalue", True),
        ("postgresql+psycopg://admin:secret@localhost:5432/db", True),
        ("SSN is 123-45-6789", True),
        ("I work with Python and PostgreSQL.", False),
    ],
)
def test_contains_sensitive_data(text, expected):
    assert contains_sensitive_data(text) is expected


def test_memory_manager_agent_process_turn_executes_structured_plan():
    semantic_service = MagicMock()
    episodic_service = MagicMock()
    target_id = uuid4()
    new_sem_id = uuid4()
    ep_id = uuid4()

    semantic_service.upsert_memory.return_value = (
        SimpleNamespace(id=new_sem_id),
        "created",
    )
    semantic_service.update_memory.return_value = SimpleNamespace(id=target_id)
    episodic_service.create_memory_if_new.return_value = (
        SimpleNamespace(id=ep_id),
        True,
    )

    plan = MemoryManagementPlan(
        decision="both",
        semantic_actions=[
            SemanticMemoryAction(
                action="create",
                content="User builds LLM assistants with LangChain.",
                category="project",
                confidence=0.9,
                is_factual=True,
            ),
            SemanticMemoryAction(
                action="create",
                content="User probably likes dark mode.",
                category="preference",
                confidence=0.4,
                is_factual=False,  # Assumption must be skipped.
            ),
            SemanticMemoryAction(
                action="update",
                target_memory_id=str(target_id),
                content="User prefers uv over pip.",
                category="preference",
                confidence=0.95,
                is_factual=True,
            ),
        ],
        episodic_actions=[
            EpisodicMemoryAction(
                summary="Designed the unified memory management pipeline.",
                event_type="decision",
                is_significant=True,
            ),
            EpisodicMemoryAction(
                summary="User said hello.",
                event_type="other",
                is_significant=False,
            ),
        ],
    )

    fake_structured_model = MagicMock()
    fake_structured_model.invoke.return_value = plan
    fake_model = MagicMock()
    fake_model.with_structured_output.return_value = fake_structured_model

    agent = MemoryManagerAgent(model=fake_model)
    report = agent.process_turn(
        user_id="user-1",
        session_id="sess-1",
        user_message="I build LLM assistants with LangChain and prefer uv over pip.",
        assistant_message="Got it!",
        semantic_service=semantic_service,
        episodic_service=episodic_service,
    )

    assert report.decision == "both"
    assert len(report.results) == 3
    assert report.skipped == [
        "non_factual_semantic_candidate",
        "insignificant_episodic_candidate",
    ]
    assert report.errors == []
    fake_model.with_structured_output.assert_called_once_with(MemoryManagementPlan)


def test_memory_manager_agent_requires_valid_user_and_session():
    agent = MemoryManagerAgent(model=MagicMock())
    with pytest.raises(ValueError, match="User ID cannot be empty"):
        agent.process_turn(
            user_id="  ",
            session_id="sess-1",
            user_message="hi",
            assistant_message="hello",
            semantic_service=MagicMock(),
            episodic_service=MagicMock(),
        )
    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        agent.process_turn(
            user_id="user-1",
            session_id="",
            user_message="hi",
            assistant_message="hello",
            semantic_service=MagicMock(),
            episodic_service=MagicMock(),
        )
