from unittest.mock import MagicMock, patch

import pytest

from app.memory.semantic import SemanticMemoryService
from app.models.semantic_memory import SemanticMemory


@pytest.fixture
def service():
    with (
        patch("app.memory.semantic.EmbeddingService") as mock_emb_cls,
        patch("app.memory.semantic.SemanticMemoryRepository") as mock_repo_cls,
    ):
        svc = SemanticMemoryService(MagicMock())
        svc._mock_embedding = mock_emb_cls.return_value
        svc._mock_repo = mock_repo_cls.return_value
        yield svc


def test_create_memory_embeds_and_persists(service):
    fake_embedding = [0.1, 0.2, 0.3]
    created = SemanticMemory(
        user_id="user-1",
        content="Likes Python and FastAPI.",
        embedding=fake_embedding,
        category="technical",
        confidence=0.9,
        source_session_id="sess-1",
        metadata_json={"source": "chat"},
    )
    service._mock_embedding.embed_document.return_value = fake_embedding
    service._mock_repo.create.return_value = created

    result = service.create_memory(
        user_id="user-1",
        content="Likes Python and FastAPI.",
        category="technical",
        confidence=0.9,
        source_session_id="sess-1",
        metadata_json={"source": "chat"},
    )

    assert result is created
    service._mock_embedding.embed_document.assert_called_once_with(
        "Likes Python and FastAPI."
    )
    service._mock_repo.create.assert_called_once_with(
        user_id="user-1",
        content="Likes Python and FastAPI.",
        embedding=fake_embedding,
        category="technical",
        confidence=0.9,
        source_session_id="sess-1",
        metadata_json={"source": "chat"},
    )


@pytest.mark.parametrize("invalid_user_id", ["", "   ", "\t\n"])
def test_create_memory_empty_user_id_raises_value_error(service, invalid_user_id):
    with pytest.raises(ValueError, match="User ID cannot be empty"):
        service.create_memory(user_id=invalid_user_id, content="Valid content")

    service._mock_embedding.embed_document.assert_not_called()
    service._mock_repo.create.assert_not_called()


@pytest.mark.parametrize("invalid_content", ["", "   ", "\n\t"])
def test_create_memory_empty_content_raises_value_error(service, invalid_content):
    with pytest.raises(ValueError, match="Content cannot be empty"):
        service.create_memory(user_id="user-1", content=invalid_content)

    service._mock_embedding.embed_document.assert_not_called()
    service._mock_repo.create.assert_not_called()


@pytest.mark.parametrize("invalid_confidence", [-0.01, 1.01, 2.0, True])
def test_create_memory_invalid_confidence_raises_value_error(service, invalid_confidence):
    with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
        service.create_memory(
            user_id="user-1",
            content="Valid content",
            confidence=invalid_confidence,
        )

    service._mock_embedding.embed_document.assert_not_called()
    service._mock_repo.create.assert_not_called()


@pytest.mark.parametrize("valid_confidence", [0.0, 0.5, 1.0])
def test_create_memory_boundary_confidence_succeeds(service, valid_confidence):
    service._mock_embedding.embed_document.return_value = [0.5]
    service._mock_repo.create.return_value = MagicMock()

    service.create_memory(
        user_id="user-1",
        content="Valid content",
        confidence=valid_confidence,
    )

    service._mock_embedding.embed_document.assert_called_once_with("Valid content")
    service._mock_repo.create.assert_called_once()


def test_search_memories_embeds_query_and_returns_results(service):
    query_embedding = [0.2, 0.4, 0.6]
    expected_results = [
        SemanticMemory(user_id="user-1", content="Uses FastAPI", embedding=query_embedding)
    ]
    service._mock_embedding.embed_query.return_value = query_embedding
    service._mock_repo.search_similar.return_value = expected_results

    results = service.search_memories(
        user_id="user-1",
        query="What frameworks does the user know?",
        top_k=3,
        max_distance=0.45,
    )

    assert results == expected_results
    service._mock_embedding.embed_query.assert_called_once_with(
        "What frameworks does the user know?"
    )
    service._mock_repo.search_similar.assert_called_once_with(
        user_id="user-1",
        query_embedding=query_embedding,
        top_k=3,
        max_distance=0.45,
    )


@pytest.mark.parametrize("invalid_user_id", ["", "   ", "\t\n"])
def test_search_memories_empty_user_id_raises_value_error(service, invalid_user_id):
    with pytest.raises(ValueError, match="User ID cannot be empty"):
        service.search_memories(user_id=invalid_user_id, query="Valid query")

    service._mock_embedding.embed_query.assert_not_called()
    service._mock_repo.search_similar.assert_not_called()


@pytest.mark.parametrize("invalid_query", ["", "   ", "\n\t"])
def test_search_memories_empty_query_raises_value_error(service, invalid_query):
    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.search_memories(user_id="user-1", query=invalid_query)

    service._mock_embedding.embed_query.assert_not_called()
    service._mock_repo.search_similar.assert_not_called()


@pytest.mark.parametrize("invalid_top_k", [0, -1, True])
def test_search_memories_invalid_top_k_raises_value_error(service, invalid_top_k):
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        service.search_memories(user_id="user-1", query="Valid query", top_k=invalid_top_k)

    service._mock_embedding.embed_query.assert_not_called()
    service._mock_repo.search_similar.assert_not_called()


@pytest.mark.parametrize("invalid_max_distance", [-0.01, 2.01, 3.0, True])
def test_search_memories_invalid_max_distance_raises_value_error(
    service, invalid_max_distance
):
    with pytest.raises(ValueError, match="max_distance must be between 0.0 and 2.0"):
        service.search_memories(
            user_id="user-1",
            query="Valid query",
            max_distance=invalid_max_distance,
        )

    service._mock_embedding.embed_query.assert_not_called()
    service._mock_repo.search_similar.assert_not_called()


@pytest.mark.parametrize("valid_max_distance", [0.0, 1.0, 2.0])
def test_search_memories_boundary_max_distance_succeeds(service, valid_max_distance):
    service._mock_embedding.embed_query.return_value = [0.5]
    service._mock_repo.search_similar.return_value = []

    result = service.search_memories(
        user_id="user-1",
        query="Valid query",
        max_distance=valid_max_distance,
    )

    assert result == []
    service._mock_embedding.embed_query.assert_called_once_with("Valid query")
    service._mock_repo.search_similar.assert_called_once_with(
        user_id="user-1",
        query_embedding=[0.5],
        top_k=5,
        max_distance=valid_max_distance,
    )
