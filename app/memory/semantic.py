from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.semantic_memory import SemanticMemory
from app.repositories.semantic_memory_repository import SemanticMemoryRepository
from app.services.embedding_service import EmbeddingService


class SemanticMemoryService:
    """
    Coordinates embedding generation, lifecycle, and persistence for semantic memories.
    """

    def __init__(self, db: Session):
        self.embedding_service = EmbeddingService()
        self.repository = SemanticMemoryRepository(db)

    def _validate_non_empty_text(self, value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    def _validate_optional_non_empty_text(
        self,
        value: str | None,
        field_name: str,
    ) -> None:
        if value is not None:
            self._validate_non_empty_text(value, field_name)

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

    @staticmethod
    def _validate_min_int(value: int, min_value: int, message: str) -> None:
        if type(value) is not int or value < min_value:
            raise ValueError(message)

    @staticmethod
    def _validate_uuid(memory_id: UUID) -> None:
        if not isinstance(memory_id, UUID):
            raise ValueError("Memory ID must be a valid UUID")

    @staticmethod
    def _validate_optional_dict(value: dict[str, Any] | None, field_name: str) -> None:
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a dictionary")

    def _build_filter_kwargs(
        self,
        category: str | None,
        metadata_filters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self._validate_optional_non_empty_text(category, "Category")
        self._validate_optional_dict(metadata_filters, "Metadata filters")
        kwargs: dict[str, Any] = {}
        if category is not None:
            kwargs["category"] = category
        if metadata_filters is not None:
            kwargs["metadata_filters"] = metadata_filters
        return kwargs

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
        self._validate_optional_non_empty_text(category, "Category")
        self._validate_float_range(confidence, 0.0, 1.0, "Confidence")
        self._validate_optional_non_empty_text(source_session_id, "Session ID")
        self._validate_optional_dict(metadata_json, "Metadata")

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

    def get_memory(
        self,
        memory_id: UUID,
        user_id: str,
    ) -> SemanticMemory | None:
        """
        Retrieves a single semantic memory by ID scoped to the owning user.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_uuid(memory_id)
        return self.repository.get_by_id(memory_id=memory_id, user_id=user_id)

    def get_user_memories(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        active_only: bool = True,
        category: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SemanticMemory]:
        """
        Retrieves paginated semantic memories belonging to a user.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_min_int(limit, 1, "limit must be greater than zero")
        self._validate_min_int(offset, 0, "offset cannot be negative")
        filter_kwargs = self._build_filter_kwargs(category, metadata_filters)

        return self.repository.get_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
            active_only=active_only,
            **filter_kwargs,
        )

    def search_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_distance: float | None = None,
        category: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SemanticMemory]:
        """
        Validates inputs, generates a query embedding, and retrieves similar memories.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(query, "Query")
        self._validate_min_int(top_k, 1, "top_k must be greater than zero")
        self._validate_float_range(max_distance, 0.0, 2.0, "max_distance")
        filter_kwargs = self._build_filter_kwargs(category, metadata_filters)

        query_embedding = self.embedding_service.embed_query(query)

        return self.repository.search_similar(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            max_distance=max_distance,
            **filter_kwargs,
        )

    def search_memories_with_distance(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        max_distance: float | None = None,
        category: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[tuple[SemanticMemory, float]]:
        """
        Retrieve semantically similar active memories with their cosine distances.

        Lower distance values represent closer semantic matches.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(query, "Query")
        self._validate_min_int(top_k, 1, "top_k must be greater than zero")
        self._validate_float_range(max_distance, 0.0, 2.0, "max_distance")
        filter_kwargs = self._build_filter_kwargs(category, metadata_filters)

        query_embedding = self.embedding_service.embed_query(query)
        return self.repository.search_similar_with_distance(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=top_k,
            max_distance=max_distance,
            **filter_kwargs,
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
        self._validate_uuid(memory_id)
        if content is not None:
            self._validate_non_empty_text(content, "Content")
        self._validate_optional_non_empty_text(category, "Category")
        self._validate_float_range(confidence, 0.0, 1.0, "Confidence")
        self._validate_optional_dict(metadata_json, "Metadata")

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

    def deactivate_memory(
        self,
        memory_id: UUID,
        user_id: str,
    ) -> SemanticMemory | None:
        """
        Soft-deletes a semantic memory while keeping it auditable and idempotent.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_uuid(memory_id)

        existing_memory = self.repository.get_by_id(memory_id, user_id)
        if existing_memory is None:
            return None
        if not existing_memory.is_active:
            return existing_memory

        return self.repository.deactivate(memory_id=memory_id, user_id=user_id)

    def upsert_memory(
        self,
        user_id: str,
        content: str,
        category: str | None = None,
        confidence: float | None = None,
        source_session_id: str | None = None,
        metadata_json: dict | None = None,
        duplicate_distance_threshold: float | None = 0.08,
    ) -> tuple[SemanticMemory, str]:
        """
        Creates a semantic memory or updates an existing duplicate cleanly.

        Returns `(memory, status)` where `status` is `'created'`, `'updated'`,
        or `'unchanged'`.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(content, "Content")
        self._validate_optional_non_empty_text(category, "Category")
        self._validate_float_range(confidence, 0.0, 1.0, "Confidence")
        self._validate_optional_non_empty_text(source_session_id, "Session ID")
        self._validate_optional_dict(metadata_json, "Metadata")
        self._validate_float_range(
            duplicate_distance_threshold,
            0.0,
            2.0,
            "duplicate_distance_threshold",
        )

        exact_match = self.repository.find_active_by_content(
            user_id=user_id,
            content=content,
        )
        if exact_match is not None:
            needs_metadata_update = (
                (category is not None and category != exact_match.category)
                or (confidence is not None and confidence != exact_match.confidence)
                or (
                    metadata_json is not None
                    and metadata_json != exact_match.metadata_json
                )
            )
            if needs_metadata_update:
                updated = self.repository.update(
                    memory_id=exact_match.id,
                    user_id=user_id,
                    category=category,
                    confidence=confidence,
                    metadata_json=metadata_json,
                    memory=exact_match,
                )
                return (updated or exact_match), "updated"
            return exact_match, "unchanged"

        embedding = self.embedding_service.embed_document(content)

        if duplicate_distance_threshold is not None:
            similar = self.repository.search_similar_with_distance(
                user_id=user_id,
                query_embedding=embedding,
                top_k=1,
                max_distance=duplicate_distance_threshold,
            )
            if similar:
                near_match, _distance = similar[0]
                updated = self.repository.update(
                    memory_id=near_match.id,
                    user_id=user_id,
                    content=content,
                    embedding=embedding if content != near_match.content else None,
                    category=category,
                    confidence=confidence,
                    metadata_json=metadata_json,
                    memory=near_match,
                )
                return (updated or near_match), "updated"

        created = self.repository.create(
            user_id=user_id,
            content=content,
            embedding=embedding,
            category=category,
            confidence=confidence,
            source_session_id=source_session_id,
            metadata_json=metadata_json,
        )
        return created, "created"
