from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.memory.episodic import EpisodicMemoryService
from app.memory.semantic import SemanticMemoryService
from app.models.episodic_memory import EpisodicMemory
from app.models.semantic_memory import SemanticMemory


@dataclass(slots=True)
class MemoryRetrievalResult:
    """Memories retrieved from the application's long-term memory stores."""

    semantic_memories: list[tuple[SemanticMemory, float]]
    episodic_memories: list[EpisodicMemory]

    @property
    def total_count(self) -> int:
        """Return the total number of retrieved memories."""
        return len(self.semantic_memories) + len(self.episodic_memories)


class MemoryRetrievalService:
    """
    Coordinates retrieval across semantic and episodic memory services.

    Semantic memories are retrieved by vector similarity. Episodic memories
    are retrieved by recency because the current episodic schema does not
    contain embeddings yet.
    """

    def __init__(self, db: Session):
        self.semantic_service = SemanticMemoryService(db)
        self.episodic_service = EpisodicMemoryService(db)

    @staticmethod
    def _validate_non_empty_text(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be greater than zero")

    @staticmethod
    def _validate_optional_distance(max_distance: float | None) -> None:
        if max_distance is None:
            return
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, (int, float))
            or not 0.0 <= max_distance <= 2.0
        ):
            raise ValueError("max_distance must be between 0.0 and 2.0")

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        semantic_limit: int = 5,
        episodic_limit: int = 5,
        max_distance: float | None = None,
        category: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        event_type: str | None = None,
    ) -> MemoryRetrievalResult:
        """
        Retrieve user-scoped semantic matches and recent episodic events.

        The two stores use different retrieval strategies:
        - Semantic: embedding similarity to the supplied query (with optional
          cosine-distance threshold, category, and metadata filters).
        - Episodic: most recent events belonging to the user (optionally scoped
          to a session or event type).
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(query, "Query")
        self._validate_positive_int(semantic_limit, "semantic_limit")
        self._validate_positive_int(episodic_limit, "episodic_limit")
        self._validate_optional_distance(max_distance)

        if category is not None:
            self._validate_non_empty_text(category, "Category")
        if session_id is not None:
            self._validate_non_empty_text(session_id, "Session ID")
        if event_type is not None:
            self._validate_non_empty_text(event_type, "Event type")
        if metadata_filters is not None and not isinstance(metadata_filters, dict):
            raise ValueError("Metadata filters must be a dictionary")

        semantic_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "query": query,
            "top_k": semantic_limit,
        }
        if max_distance is not None:
            semantic_kwargs["max_distance"] = max_distance
        if category is not None:
            semantic_kwargs["category"] = category
        if metadata_filters is not None:
            semantic_kwargs["metadata_filters"] = metadata_filters

        semantic_memories = self.semantic_service.search_memories_with_distance(
            **semantic_kwargs
        )

        if session_id is not None:
            session_memories = self.episodic_service.get_session_memories(
                user_id=user_id,
                session_id=session_id,
            )
            if event_type is not None:
                session_memories = [
                    m for m in session_memories if m.event_type == event_type
                ]
            episodic_memories = session_memories[:episodic_limit]
        else:
            episodic_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "limit": episodic_limit,
            }
            if event_type is not None:
                episodic_kwargs["event_type"] = event_type
            if metadata_filters is not None:
                episodic_kwargs["metadata_filters"] = metadata_filters
            episodic_memories = self.episodic_service.get_user_memories(
                **episodic_kwargs
            )

        return MemoryRetrievalResult(
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
        )
