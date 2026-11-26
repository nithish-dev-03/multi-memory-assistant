
from sqlalchemy.orm import Session, object_session
from sqlalchemy import select
import uuid

from app.models.semantic_memory import SemanticMemory


class SemanticMemoryRepository:
    """
    Handles database operations for semantic memories.
    """

    def __init__(self, db: Session):
        self.db = db

    def _validate_min_int(self, value: int, min_value: int, message: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < min_value:
            raise ValueError(message)

    def create(
        self,
        user_id: str,
        content: str,
        embedding: list[float],
        category: str | None = None,
        confidence: float | None = None,
        source_session_id: str | None = None,
        metadata_json: dict | None = None,
    ) -> SemanticMemory:
        """
        Creates and persists a semantic memory.
        """

        memory = SemanticMemory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            category=category,
            confidence=confidence,
            source_session_id=source_session_id,
            metadata_json=metadata_json,
        )

        try:
            self.db.add(memory)
            self.db.commit()
            self.db.refresh(memory)

            return memory

        except Exception:
            self.db.rollback()
            raise

    def search_similar(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        max_distance: float | None = None,
    ) -> list[SemanticMemory]:
        """
        Retrieves the most semantically similar active memories
        belonging to a specific user.
        """

        self._validate_min_int(top_k, 1, "top_k must be greater than zero")

        distance = SemanticMemory.embedding.cosine_distance(
            query_embedding
        )

        statement = (
            select(SemanticMemory)
            .where(
                SemanticMemory.user_id == user_id,
                SemanticMemory.is_active.is_(True),
            )
            .order_by(distance.asc())
            .limit(top_k)
        )

        if max_distance is not None:
            statement = statement.where(distance <= max_distance)

        return list(self.db.scalars(statement).all())
    
    def get_by_id(
        self,
        memory_id: uuid.UUID,
        user_id: str,
    ) -> SemanticMemory | None:

        statement = select(SemanticMemory).where(
            SemanticMemory.id == memory_id,
            SemanticMemory.user_id == user_id,
        )

        return self.db.scalar(statement)

    
    def get_by_user(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
        active_only: bool = True,
    ) -> list[SemanticMemory]:

        self._validate_min_int(limit, 1, "limit must be greater than zero")
        self._validate_min_int(offset, 0, "offset cannot be negative")

        statement = select(SemanticMemory).where(
            SemanticMemory.user_id == user_id
        )

        if active_only:
            statement = statement.where(
                SemanticMemory.is_active.is_(True)
            )

        statement = (
            statement
            .order_by(
                SemanticMemory.created_at.desc(),
                SemanticMemory.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )

        return list(self.db.scalars(statement).all())

    def deactivate(
        self,
        memory_id: uuid.UUID,
        user_id: str,
    ) -> SemanticMemory | None:

        memory = self.get_by_id(memory_id, user_id)

        if memory is None:
            return None

        try:
            memory.is_active = False

            self.db.commit()
            self.db.refresh(memory)

            return memory

        except Exception:
            self.db.rollback()
            raise


    def update(
        self,
        memory_id: uuid.UUID,
        user_id: str,
        content: str | None = None,
        embedding: list[float] | None = None,
        category: str | None = None,
        confidence: float | None = None,
        metadata_json: dict | None = None,
        memory: SemanticMemory | None = None,
    ) -> SemanticMemory | None:
        if memory is None:
            memory = self.get_by_id(memory_id, user_id)
        else:
            if memory.id != memory_id or memory.user_id != user_id:
                raise ValueError("Provided memory does not match memory_id and user_id")
            if isinstance(self.db, Session) and object_session(memory) is not self.db:
                raise ValueError(
                    "Provided memory does not belong to the current database session"
                )

        if memory is None:
            return None

        if content is not None and not content.strip():
            raise ValueError("Content cannot be empty")

        content_changed = content is not None and content != memory.content

        if embedding is not None and not content_changed:
            raise ValueError("Embedding can only be updated when content changes")

        if content_changed and embedding is None:
            raise ValueError("Embedding is required when content changes")

        if content_changed:
            memory.content = content
            memory.embedding = embedding

        if category is not None:
            memory.category = category

        if confidence is not None:
            memory.confidence = confidence

        if metadata_json is not None:
            memory.metadata_json = metadata_json

        try:
            self.db.commit()
            self.db.refresh(memory)
            return memory

        except Exception:
            self.db.rollback()
            raise