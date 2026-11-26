import uuid
from unittest.mock import MagicMock

import pytest

from app.models.semantic_memory import SemanticMemory
from app.repositories.semantic_memory_repository import (
    SemanticMemoryRepository,
)


@pytest.mark.parametrize(
    ("limit", "offset", "expected_message"),
    [
        (0, 0, "limit must be greater than zero"),
        (-5, 0, "limit must be greater than zero"),
        (10, -1, "offset cannot be negative"),
    ],
)
def test_get_by_user_invalid_pagination_raises_value_error(
    limit: int,
    offset: int,
    expected_message: str,
):
    repository = SemanticMemoryRepository(MagicMock())

    with pytest.raises(ValueError, match=expected_message):
        repository.get_by_user(
            user_id="test-user-001",
            limit=limit,
            offset=offset,
        )


def test_get_by_user_orders_by_created_at_desc_then_id_desc():
    mock_db = MagicMock()
    mock_db.scalars.return_value.all.return_value = []
    repository = SemanticMemoryRepository(mock_db)

    repository.get_by_user(user_id="test-user-001", limit=10, offset=5)

    statement = mock_db.scalars.call_args.args[0]
    compiled_sql = str(statement)
    assert "ORDER BY semantic_memories.created_at DESC, semantic_memories.id DESC" in compiled_sql


def test_get_by_user_filters_active_only_when_enabled():
    mock_db = MagicMock()
    mock_db.scalars.return_value.all.return_value = []
    repository = SemanticMemoryRepository(mock_db)

    repository.get_by_user(user_id="test-user-001", active_only=True)
    active_sql = str(mock_db.scalars.call_args.args[0])

    repository.get_by_user(user_id="test-user-001", active_only=False)
    all_sql = str(mock_db.scalars.call_args.args[0])

    assert "semantic_memories.is_active IS true" in active_sql
    assert "semantic_memories.is_active IS true" not in all_sql


def test_get_by_id_scopes_query_to_memory_and_user():
    mock_db = MagicMock()
    expected = SemanticMemory(user_id="test-user-001", content="fact", embedding=[0.1])
    mock_db.scalar.return_value = expected
    repository = SemanticMemoryRepository(mock_db)
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")

    result = repository.get_by_id(memory_id=memory_id, user_id="test-user-001")

    assert result is expected
    statement = mock_db.scalar.call_args.args[0]
    compiled_sql = str(statement)
    assert "semantic_memories.id = :id_1" in compiled_sql
    assert "semantic_memories.user_id = :user_id_1" in compiled_sql


def test_search_similar_invalid_top_k_raises_value_error():
    repository = SemanticMemoryRepository(MagicMock())

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        repository.search_similar(user_id="test-user-001", query_embedding=[0.1], top_k=0)


def test_deactivate_existing_user_memory_sets_is_active_false():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="fact",
        embedding=[0.1],
        is_active=True,
    )
    mock_db.scalar.return_value = memory
    repository = SemanticMemoryRepository(mock_db)

    result = repository.deactivate(memory_id=memory_id, user_id="test-user-001")

    assert result is memory
    assert result.is_active is False
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(memory)


def test_deactivate_nonexistent_memory_returns_none():
    mock_db = MagicMock()
    mock_db.scalar.return_value = None
    repository = SemanticMemoryRepository(mock_db)

    result = repository.deactivate(memory_id=uuid.uuid4(), user_id="test-user-001")

    assert result is None
    mock_db.commit.assert_not_called()
    mock_db.refresh.assert_not_called()


def test_deactivate_memory_belonging_to_another_user_returns_none():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    other_users_memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="fact",
        embedding=[0.1],
        is_active=True,
    )
    # Scoped lookup (id == memory_id AND user_id == "other-user-999") yields no row.
    mock_db.scalar.return_value = None
    repository = SemanticMemoryRepository(mock_db)

    result = repository.deactivate(memory_id=memory_id, user_id="other-user-999")

    assert result is None
    assert other_users_memory.is_active is True
    statement = mock_db.scalar.call_args.args[0]
    compiled_sql = str(statement)
    assert "semantic_memories.id = :id_1" in compiled_sql
    assert "semantic_memories.user_id = :user_id_1" in compiled_sql
    mock_db.commit.assert_not_called()


def test_update_content_without_embedding_raises_value_error():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="original content",
        embedding=[0.1],
    )
    mock_db.scalar.return_value = memory
    repository = SemanticMemoryRepository(mock_db)

    with pytest.raises(ValueError, match="Embedding is required when content changes"):
        repository.update(
            memory_id=memory_id,
            user_id="test-user-001",
            content="updated content",
        )

    assert memory.content == "original content"
    mock_db.commit.assert_not_called()


def test_update_content_with_new_embedding_updates_both():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="original content",
        embedding=[0.1],
    )
    mock_db.scalar.return_value = memory
    repository = SemanticMemoryRepository(mock_db)

    result = repository.update(
        memory_id=memory_id,
        user_id="test-user-001",
        content="updated content",
        embedding=[0.9],
    )

    assert result is memory
    assert memory.content == "updated content"
    assert memory.embedding == [0.9]
    mock_db.commit.assert_called_once()
    mock_db.refresh.assert_called_once_with(memory)


def test_update_confidence_only_does_not_require_new_embedding():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    original_embedding = [0.1, 0.2]
    memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="original content",
        embedding=original_embedding,
        confidence=0.5,
    )
    mock_db.scalar.return_value = memory
    repository = SemanticMemoryRepository(mock_db)

    result = repository.update(
        memory_id=memory_id,
        user_id="test-user-001",
        confidence=0.95,
    )

    assert result is memory
    assert memory.confidence == 0.95
    assert memory.embedding == original_embedding
    mock_db.commit.assert_called_once()


def test_update_category_only_does_not_require_new_embedding():
    mock_db = MagicMock()
    memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
    original_embedding = [0.1, 0.2]
    memory = SemanticMemory(
        id=memory_id,
        user_id="test-user-001",
        content="original content",
        embedding=original_embedding,
        category="general",
    )
    mock_db.scalar.return_value = memory
    repository = SemanticMemoryRepository(mock_db)

    result = repository.update(
        memory_id=memory_id,
        user_id="test-user-001",
        category="professional",
    )

    assert result is memory
    assert memory.category == "professional"
    assert memory.embedding == original_embedding
    mock_db.commit.assert_called_once()


def test_update_memory_belonging_to_another_user_returns_none():
    mock_db = MagicMock()
    mock_db.scalar.return_value = None
    repository = SemanticMemoryRepository(mock_db)

    result = repository.update(
        memory_id=uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53"),
        user_id="other-user-999",
        confidence=0.99,
    )

    assert result is None
    mock_db.commit.assert_not_called()
