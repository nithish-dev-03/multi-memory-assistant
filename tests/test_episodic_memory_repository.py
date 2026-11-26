import uuid
from unittest.mock import MagicMock

import pytest

from app.models.episodic_memory import EpisodicMemory
from app.repositories.episodic_memory_repository import EpisodicMemoryRepository


def test_create_persists_episodic_memory_with_optional_fields():
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.create(
        user_id="user-1",
        summary="User discussed migrating API endpoints to FastAPI.",
        session_id="sess-1",
        event_type="conversation_summary",
        metadata_json={"topic": "fastapi"},
    )

    assert isinstance(result, EpisodicMemory)
    assert result.user_id == "user-1"
    assert result.summary == "User discussed migrating API endpoints to FastAPI."
    assert result.session_id == "sess-1"
    assert result.event_type == "conversation_summary"
    assert result.metadata_json == {"topic": "fastapi"}
    mock_db.add.assert_called_once_with(result)
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(result)


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"user_id": "", "summary": "Valid summary"}, "User ID cannot be empty"),
        ({"user_id": "   ", "summary": "Valid summary"}, "User ID cannot be empty"),
        ({"user_id": "user-1", "summary": ""}, "Summary cannot be empty"),
        ({"user_id": "user-1", "summary": "   "}, "Summary cannot be empty"),
        (
            {"user_id": "user-1", "summary": "Valid", "session_id": "   "},
            "Session ID cannot be empty",
        ),
        (
            {"user_id": "user-1", "summary": "Valid", "event_type": ""},
            "Event type cannot be empty",
        ),
        (
            {"user_id": "user-1", "summary": "Valid", "metadata_json": ["bad"]},
            "Metadata must be a dictionary",
        ),
    ],
)
def test_create_rejects_invalid_inputs(kwargs, expected_message):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match=expected_message):
        repository.create(**kwargs)

    mock_db.add.assert_not_called()
    mock_db.commit.assert_not_called()


def test_create_rolls_back_on_commit_error():
    mock_db = MagicMock()
    mock_db.commit.side_effect = RuntimeError("db failure")
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(RuntimeError, match="db failure"):
        repository.create(user_id="user-1", summary="Valid summary")

    mock_db.rollback.assert_called_once()


def test_get_by_id_existing_memory_and_correct_user_returns_memory():
    mock_db = MagicMock()
    memory_id = uuid.uuid4()
    expected = EpisodicMemory(
        id=memory_id,
        user_id="user-1",
        summary="Completed onboarding.",
    )
    mock_db.execute.return_value.scalar_one_or_none.return_value = expected
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_id(memory_id=memory_id, user_id="user-1")

    assert result is expected
    statement = mock_db.execute.call_args.args[0]
    compiled_sql = str(statement)
    assert "episodic_memories.id = :id_1" in compiled_sql
    assert "episodic_memories.user_id = :user_id_1" in compiled_sql


def test_get_by_id_nonexistent_memory_returns_none():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_id(memory_id=uuid.uuid4(), user_id="user-1")

    assert result is None


def test_get_by_id_memory_belonging_to_another_user_returns_none():
    mock_db = MagicMock()
    memory_id = uuid.uuid4()
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_id(memory_id=memory_id, user_id="other-user")

    assert result is None
    statement = mock_db.execute.call_args.args[0]
    assert statement.compile().params["user_id_1"] == "other-user"


@pytest.mark.parametrize("invalid_user_id", ["", "   ", "\t\n"])
def test_get_by_id_empty_user_id_raises_value_error(invalid_user_id):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="User ID cannot be empty"):
        repository.get_by_id(memory_id=uuid.uuid4(), user_id=invalid_user_id)

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_memory_id", ["not-a-uuid", str(uuid.uuid4()), 123, None])
def test_get_by_id_invalid_memory_id_type_raises_value_error(invalid_memory_id):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="Memory ID must be a valid UUID"):
        repository.get_by_id(memory_id=invalid_memory_id, user_id="user-1")

    mock_db.execute.assert_not_called()


def test_get_by_session_returns_user_session_memories():
    mock_db = MagicMock()
    expected = [
        EpisodicMemory(id=uuid.uuid4(), user_id="user-1", session_id="sess-1", summary="Second"),
        EpisodicMemory(id=uuid.uuid4(), user_id="user-1", session_id="sess-1", summary="First"),
    ]
    mock_db.execute.return_value.scalars.return_value.all.return_value = expected
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_session(user_id="user-1", session_id="sess-1")

    assert result == expected
    statement = mock_db.execute.call_args.args[0]
    params = statement.compile().params
    assert params["user_id_1"] == "user-1"
    assert params["session_id_1"] == "sess-1"


def test_get_by_session_empty_returns_empty_list():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_session(user_id="user-1", session_id="empty-sess")

    assert result == []


def test_get_by_session_excludes_other_user_memories():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_session(user_id="user-1", session_id="shared-sess")

    assert result == []
    statement = mock_db.execute.call_args.args[0]
    compiled = statement.compile()
    assert "episodic_memories.user_id = :user_id_1" in str(compiled)
    assert compiled.params["user_id_1"] == "user-1"


def test_get_by_session_excludes_other_session_memories():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_session(user_id="user-1", session_id="sess-1")

    assert result == []
    statement = mock_db.execute.call_args.args[0]
    compiled = statement.compile()
    assert "episodic_memories.session_id = :session_id_1" in str(compiled)
    assert compiled.params["session_id_1"] == "sess-1"


@pytest.mark.parametrize("invalid_user_id", ["", "   ", "\t\n"])
def test_get_by_session_empty_user_id_raises_value_error(invalid_user_id):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="User ID cannot be empty"):
        repository.get_by_session(user_id=invalid_user_id, session_id="sess-1")

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_session_id", ["", "   ", "\t\n"])
def test_get_by_session_empty_session_id_raises_value_error(invalid_session_id):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        repository.get_by_session(user_id="user-1", session_id=invalid_session_id)

    mock_db.execute.assert_not_called()


def test_get_by_session_orders_by_created_at_desc_then_id_desc():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    repository.get_by_session(user_id="user-1", session_id="sess-1")

    statement = mock_db.execute.call_args.args[0]
    assert "ORDER BY episodic_memories.created_at DESC, episodic_memories.id DESC" in str(
        statement
    )


def test_get_by_user_returns_only_requested_user_memories():
    mock_db = MagicMock()
    expected = [
        EpisodicMemory(id=uuid.uuid4(), user_id="user-1", summary="Recent event"),
    ]
    mock_db.execute.return_value.scalars.return_value.all.return_value = expected
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_user(user_id="user-1")

    assert result == expected
    statement = mock_db.execute.call_args.args[0]
    compiled = statement.compile()
    assert "WHERE episodic_memories.user_id = :user_id_1" in str(compiled)
    assert compiled.params["user_id_1"] == "user-1"


def test_get_by_user_applies_limit():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    repository.get_by_user(user_id="user-1", limit=5)

    statement = mock_db.execute.call_args.args[0]
    compiled = statement.compile()
    assert "LIMIT :param_1" in str(compiled)
    assert compiled.params["param_1"] == 5


def test_get_by_user_applies_offset():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    repository.get_by_user(user_id="user-1", limit=10, offset=25)

    statement = mock_db.execute.call_args.args[0]
    compiled = statement.compile()
    assert "OFFSET :param_2" in str(compiled)
    assert compiled.params["param_1"] == 10
    assert compiled.params["param_2"] == 25


def test_get_by_user_empty_returns_empty_list():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    result = repository.get_by_user(user_id="user-1")

    assert result == []


@pytest.mark.parametrize("invalid_user_id", ["", "   ", "\t\n"])
def test_get_by_user_empty_user_id_raises_value_error(invalid_user_id):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="User ID cannot be empty"):
        repository.get_by_user(user_id=invalid_user_id)

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_limit", [0, -1, -10])
def test_get_by_user_zero_or_negative_limit_raises_value_error(invalid_limit):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        repository.get_by_user(user_id="user-1", limit=invalid_limit)

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_limit", [True, False, 1.5, 2.0])
def test_get_by_user_boolean_or_float_limit_raises_value_error(invalid_limit):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        repository.get_by_user(user_id="user-1", limit=invalid_limit)

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_offset", [-1, -10])
def test_get_by_user_negative_offset_raises_value_error(invalid_offset):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="offset cannot be negative"):
        repository.get_by_user(user_id="user-1", offset=invalid_offset)

    mock_db.execute.assert_not_called()


@pytest.mark.parametrize("invalid_offset", [True, False, 0.5, 1.0])
def test_get_by_user_boolean_or_float_offset_raises_value_error(invalid_offset):
    mock_db = MagicMock()
    repository = EpisodicMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="offset cannot be negative"):
        repository.get_by_user(user_id="user-1", offset=invalid_offset)

    mock_db.execute.assert_not_called()


def test_get_by_user_orders_deterministically():
    mock_db = MagicMock()
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repository = EpisodicMemoryRepository(mock_db)

    repository.get_by_user(user_id="user-1")

    statement = mock_db.execute.call_args.args[0]
    assert "ORDER BY episodic_memories.created_at DESC, episodic_memories.id DESC" in str(
        statement
    )
