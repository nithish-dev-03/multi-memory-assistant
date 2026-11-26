from sqlalchemy.orm import Session
from uuid import UUID

from app.models.semantic_memory import SemanticMemory
from app.repositories.semantic_memory_repository import SemanticMemoryRepository
from app.services.embedding_service import EmbeddingService


class SemanticMemoryService:
    """
    Coordinates embedding generation and persistence for semantic memories.
    """

    def __init__(self, db: Session):
        self.embedding_service = EmbeddingService()
        self.repository = SemanticMemoryRepository(db)

    def _validate_non_empty_text(self, value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    def _validate_float_range(
        self,
        value: float | None,
        min_value: float,
        max_value: float,
        field_name: str,
    ) -> None:
        if value is None:
            return
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not min_value <= value <= max_value
        ):
            raise ValueError(
                f"{field_name} must be between {min_value} and {max_value}"
            )

    def create_memory(
        self,
        user_id: str,
        content: str,
        category: str | None = None,
        confidence: float | None = None,
        source_session_id: str | None = None,
        metadata_json: dict | None = None,
    ) -> SemanticMemory:
        """
        Validates inputs, generates a document embedding, and persists a semantic memory.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(content, "Content")
        self._validate_float_range(confidence, 0.0, 1.0, "Confidence")

        embedding = self.embedding_service.embed_document(content)

        return self.repository.create(
            user_id=user_id,
            content=content,
            embedding=embedding,
            category=category,
            confidence=confidence,
            source_session_id=source_session_id,
            metadata_json=metadata_json,
        )

    def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_distance: float | None = None,
    ) -> list[SemanticMemory]:
        """
        Validates inputs, generates a query embedding, and retrieves similar memories.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(query, "Query")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._validate_float_range(max_distance, 0.0, 2.0, "max_distance")

        query_embedding = self.embedding_service.embed_query(query)

        return self.repository.search_similar(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            max_distance=max_distance,
        )

    def update_memory(
        self,
        memory_id: UUID,
        user_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        metadata_json: dict | None = None,
    ) -> SemanticMemory | None:
        """
        Validates updates, regenerates the embedding only if content changes,
        and persists the updated semantic memory for the owning user.
        """
        self._validate_non_empty_text(user_id, "User ID")
        if content is not None:
            self._validate_non_empty_text(content, "Content")
        self._validate_float_range(confidence, 0.0, 1.0, "Confidence")

        existing_memory = self.repository.get_by_id(memory_id, user_id)
        if existing_memory is None:
            return None

        embedding = (
            self.embedding_service.embed_document(content)
            if content is not None and content != existing_memory.content
            else None
        )

        return self.repository.update(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            embedding=embedding,
            category=category,
            confidence=confidence,
            metadata_json=metadata_json,
            memory=existing_memory,
        )
