from unittest.mock import MagicMock, patch

import pytest

from app.memory.retrieval import MemoryRetrievalService, MemoryRetrievalResult


@pytest.fixture
def service():
    with (
        patch("app.memory.retrieval.SemanticMemoryService") as semantic_cls,
        patch("app.memory.retrieval.EpisodicMemoryService") as episodic_cls,
    ):
        instance = MemoryRetrievalService(MagicMock())
        instance.semantic_service = semantic_cls.return_value
        instance.episodic_service = episodic_cls.return_value
        yield instance


def test_retrieve_memories_combines_semantic_and_recent_episodic(service):
    semantic_results = [(MagicMock(), 0.15), (MagicMock(), 0.32)]
    episodic_results = [MagicMock()]
    service.semantic_service.search_memories_with_distance.return_value = semantic_results
    service.episodic_service.get_user_memories.return_value = episodic_results

    result = service.retrieve_memories(
        user_id="user-1",
        query="What technologies does the user use?",
        semantic_limit=4,
        episodic_limit=3,
    )

    assert isinstance(result, MemoryRetrievalResult)
    assert result.semantic_memories == semantic_results
    assert result.episodic_memories == episodic_results
    assert result.total_count == 3
    service.semantic_service.search_memories_with_distance.assert_called_once_with(
        user_id="user-1",
        query="What technologies does the user use?",
        top_k=4,
    )
    service.episodic_service.get_user_memories.assert_called_once_with(
        user_id="user-1",
        limit=3,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"user_id": "", "query": "valid"}, "User ID cannot be empty"),
        ({"user_id": "  ", "query": "valid"}, "User ID cannot be empty"),
        ({"user_id": "user-1", "query": ""}, "Query cannot be empty"),
        ({"user_id": "user-1", "query": "  "}, "Query cannot be empty"),
        (
            {"user_id": "user-1", "query": "valid", "semantic_limit": 0},
            "semantic_limit must be greater than zero",
        ),
        (
            {"user_id": "user-1", "query": "valid", "semantic_limit": True},
            "semantic_limit must be greater than zero",
        ),
        (
            {"user_id": "user-1", "query": "valid", "episodic_limit": -1},
            "episodic_limit must be greater than zero",
        ),
        (
            {"user_id": "user-1", "query": "valid", "episodic_limit": 1.5},
            "episodic_limit must be greater than zero",
        ),
    ],
)
def test_retrieve_memories_validates_inputs(service, kwargs, message):
    with pytest.raises(ValueError, match=message):
        service.retrieve_memories(**kwargs)

    service.semantic_service.search_memories_with_distance.assert_not_called()
    service.episodic_service.get_user_memories.assert_not_called()


def test_retrieve_memories_supports_threshold_filters_and_session_scoping(service):
    sem_item = (MagicMock(user_id="user-1"), 0.2)
    ep_session_match = MagicMock(user_id="user-1", event_type="decision")
    ep_other_type = MagicMock(user_id="user-1", event_type="other")
    service.semantic_service.search_memories_with_distance.return_value = [sem_item]
    service.episodic_service.get_session_memories.return_value = [
        ep_other_type,
        ep_session_match,
    ]

    result = service.retrieve_memories(
        user_id="user-1",
        query="architecture decision",
        semantic_limit=3,
        episodic_limit=2,
        max_distance=0.5,
        category="project",
        metadata_filters={"env": "dev"},
        session_id="sess-42",
        event_type="decision",
    )

    assert result.semantic_memories == [sem_item]
    assert result.episodic_memories == [ep_session_match]
    service.semantic_service.search_memories_with_distance.assert_called_once_with(
        user_id="user-1",
        query="architecture decision",
        top_k=3,
        max_distance=0.5,
        category="project",
        metadata_filters={"env": "dev"},
    )
    service.episodic_service.get_session_memories.assert_called_once_with(
        user_id="user-1",
        session_id="sess-42",
    )


def test_retrieve_memories_handles_empty_results_and_invalid_filters(service):
    service.semantic_service.search_memories_with_distance.return_value = []
    service.episodic_service.get_user_memories.return_value = []

    empty = service.retrieve_memories(user_id="user-2", query="anything")
    assert empty.total_count == 0
    assert empty.semantic_memories == []
    assert empty.episodic_memories == []

    with pytest.raises(ValueError, match="max_distance must be between 0.0 and 2.0"):
        service.retrieve_memories(user_id="user-2", query="anything", max_distance=2.5)
    with pytest.raises(ValueError, match="Category cannot be empty"):
        service.retrieve_memories(user_id="user-2", query="anything", category="  ")
    with pytest.raises(ValueError, match="Metadata filters must be a dictionary"):
        service.retrieve_memories(user_id="user-2", query="anything", metadata_filters=[])
