from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.memory.episodic import EpisodicMemoryService
from app.models.episodic_memory import EpisodicMemory


@pytest.fixture
def service():
    with patch("app.memory.episodic.EpisodicMemoryRepository") as mock_repo_cls:
        svc = EpisodicMemoryService(MagicMock())
        svc._mock_repo = mock_repo_cls.return_value
        yield svc


def test_create_memory_delegates_to_repository(service):
    created = EpisodicMemory(
        id=uuid4(),
        user_id="user-1",
        summary="Completed database migration.",
        session_id="sess-1",
        event_type="milestone",
        metadata_json={"step": 1},
    )
    service._mock_repo.create.return_value = created

    result = service.create_memory(
        user_id="user-1",
        summary="Completed database migration.",
        session_id="sess-1",
        event_type="milestone",
        metadata_json={"step": 1},
    )

    assert result is created
    service._mock_repo.create.assert_called_once_with(
        user_id="user-1",
        summary="Completed database migration.",
        session_id="sess-1",
        event_type="milestone",
        metadata_json={"step": 1},
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"user_id": "", "summary": "Valid"}, "User ID cannot be empty"),
        ({"user_id": "   ", "summary": "Valid"}, "User ID cannot be empty"),
        ({"user_id": "user-1", "summary": ""}, "Summary cannot be empty"),
        ({"user_id": "user-1", "summary": "   "}, "Summary cannot be empty"),
        (
            {"user_id": "user-1", "summary": "Valid", "session_id": "  "},
            "Session ID cannot be empty",
        ),
        (
            {"user_id": "user-1", "summary": "Valid", "event_type": ""},
            "Event type cannot be empty",
        ),
        (
            {"user_id": "user-1", "summary": "Valid", "metadata_json": "bad"},
            "Metadata must be a dictionary",
        ),
    ],
)
def test_create_memory_rejects_invalid_input(service, kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        service.create_memory(**kwargs)

    service._mock_repo.create.assert_not_called()


def test_get_memory_returns_matching_memory(service):
    memory_id = UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    expected = EpisodicMemory(id=memory_id, user_id="user-1", summary="Summary")
    service._mock_repo.get_by_id.return_value = expected

    result = service.get_memory(memory_id=memory_id, user_id="user-1")

    assert result is expected
    service._mock_repo.get_by_id.assert_called_once_with(
        memory_id=memory_id,
        user_id="user-1",
    )


def test_get_memory_returns_none_when_missing(service):
    memory_id = uuid4()
    service._mock_repo.get_by_id.return_value = None

    assert service.get_memory(memory_id=memory_id, user_id="user-1") is None


@pytest.mark.parametrize(
    ("memory_id", "user_id", "expected_message"),
    [
        (uuid4(), "", "User ID cannot be empty"),
        (uuid4(), "   ", "User ID cannot be empty"),
        ("not-a-uuid", "user-1", "Memory ID must be a valid UUID"),
        (123, "user-1", "Memory ID must be a valid UUID"),
    ],
)
def test_get_memory_rejects_invalid_input(
    service, memory_id, user_id, expected_message
):
    with pytest.raises(ValueError, match=expected_message):
        service.get_memory(memory_id=memory_id, user_id=user_id)

    service._mock_repo.get_by_id.assert_not_called()


def test_get_session_memories_delegates_to_repository(service):
    expected = [
        EpisodicMemory(id=uuid4(), user_id="user-1", session_id="sess-1", summary="S1")
    ]
    service._mock_repo.get_by_session.return_value = expected

    result = service.get_session_memories(user_id="user-1", session_id="sess-1")

    assert result == expected
    service._mock_repo.get_by_session.assert_called_once_with(
        user_id="user-1",
        session_id="sess-1",
    )


@pytest.mark.parametrize(
    ("user_id", "session_id", "expected_message"),
    [
        ("", "sess-1", "User ID cannot be empty"),
        ("   ", "sess-1", "User ID cannot be empty"),
        ("user-1", "", "Session ID cannot be empty"),
        ("user-1", "   ", "Session ID cannot be empty"),
    ],
)
def test_get_session_memories_rejects_invalid_input(
    service, user_id, session_id, expected_message
):
    with pytest.raises(ValueError, match=expected_message):
        service.get_session_memories(user_id=user_id, session_id=session_id)

    service._mock_repo.get_by_session.assert_not_called()


def test_get_user_memories_delegates_with_pagination(service):
    expected = [
        EpisodicMemory(id=uuid4(), user_id="user-1", summary="Event 1")
    ]
    service._mock_repo.get_by_user.return_value = expected

    result = service.get_user_memories(user_id="user-1", limit=10, offset=5)

    assert result == expected
    service._mock_repo.get_by_user.assert_called_once_with(
        user_id="user-1",
        limit=10,
        offset=5,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"user_id": ""}, "User ID cannot be empty"),
        ({"user_id": "   "}, "User ID cannot be empty"),
        ({"user_id": "user-1", "limit": 0}, "limit must be greater than zero"),
        ({"user_id": "user-1", "limit": -1}, "limit must be greater than zero"),
        ({"user_id": "user-1", "limit": True}, "limit must be greater than zero"),
        ({"user_id": "user-1", "limit": 1.5}, "limit must be greater than zero"),
        ({"user_id": "user-1", "offset": -1}, "offset cannot be negative"),
        ({"user_id": "user-1", "offset": True}, "offset cannot be negative"),
        ({"user_id": "user-1", "offset": 0.5}, "offset cannot be negative"),
    ],
)
def test_get_user_memories_rejects_invalid_input(service, kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        service.get_user_memories(**kwargs)

    service._mock_repo.get_by_user.assert_not_called()


def test_create_memory_if_new_prevents_duplicate_writes(service):
    existing = EpisodicMemory(
        id=uuid4(),
        user_id="user-1",
        session_id="sess-1",
        summary="Completed migration",
    )
    service._mock_repo.find_recent_by_summary.return_value = existing

    memory, created = service.create_memory_if_new(
        user_id="user-1",
        summary="Completed migration",
        session_id="sess-1",
    )
    assert memory is existing
    assert created is False
    service._mock_repo.create.assert_not_called()

    service._mock_repo.find_recent_by_summary.return_value = None
    new_memory = EpisodicMemory(
        id=uuid4(),
        user_id="user-1",
        session_id="sess-1",
        summary="Added vector index",
    )
    service._mock_repo.create.return_value = new_memory
    memory2, created2 = service.create_memory_if_new(
        user_id="user-1",
        summary="Added vector index",
        session_id="sess-1",
    )
    assert memory2 is new_memory
    assert created2 is True
    service._mock_repo.create.assert_called_once()
