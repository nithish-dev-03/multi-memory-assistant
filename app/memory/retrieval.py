from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.memory.episodic import EpisodicMemoryService
from app.memory.semantic import SemanticMemoryService
from app.models.episodic_memory import EpisodicMemory
from app.models.semantic_memory import SemanticMemory


@dataclass(slots=True)
class MemoryRetrievalResult:
    """Memories retrieved from the application's long-term memory stores."""

    semantic_memories: list[SemanticMemory]
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

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        semantic_limit: int = 5,
        episodic_limit: int = 5,
    ) -> MemoryRetrievalResult:
        """
        Retrieve user-scoped semantic matches and recent episodic events.

        The two stores use different retrieval strategies:
        - Semantic: embedding similarity to the supplied query.
        - Episodic: most recent events belonging to the user.

        Episodic semantic search can be added later if embeddings are added
        to the episodic-memory schema.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(query, "Query")
        self._validate_positive_int(semantic_limit, "semantic_limit")
        self._validate_positive_int(episodic_limit, "episodic_limit")

        semantic_memories = self.semantic_service.search_memories(
            user_id=user_id,
            query=query,
            top_k=semantic_limit,
        )
        episodic_memories = self.episodic_service.get_user_memories(
            user_id=user_id,
            limit=episodic_limit,
        )

        return MemoryRetrievalResult(
            semantic_memories=semantic_memories,
            episodic_memories=episodic_memories,
        )
