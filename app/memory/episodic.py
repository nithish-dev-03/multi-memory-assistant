from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.episodic_memory import EpisodicMemory
from app.repositories.episodic_memory_repository import EpisodicMemoryRepository


class EpisodicMemoryService:
    """
    Coordinates validation and repository operations for episodic memories.
    """

    def __init__(self, db: Session):
        self.repository = EpisodicMemoryRepository(db)

    @staticmethod
    def _validate_non_empty_text(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def _validate_optional_non_empty_text(
        value: str | None,
        field_name: str,
    ) -> None:
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def _validate_min_int(value: int, min_value: int, message: str) -> None:
        if type(value) is not int or value < min_value:
            raise ValueError(message)

    def create_memory(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        event_type: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> EpisodicMemory:
        """
        Validates inputs and creates an episodic memory.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(summary, "Summary")
        self._validate_optional_non_empty_text(session_id, "Session ID")
        self._validate_optional_non_empty_text(event_type, "Event type")

        if metadata_json is not None and not isinstance(metadata_json, dict):
            raise ValueError("Metadata must be a dictionary")

        return self.repository.create(
            user_id=user_id,
            summary=summary,
            session_id=session_id,
            event_type=event_type,
            metadata_json=metadata_json,
        )

    def get_memory(
        self,
        memory_id: UUID,
        user_id: str,
    ) -> EpisodicMemory | None:
        """
        Retrieves a single episodic memory by ID scoped to the owning user.
        """
        self._validate_non_empty_text(user_id, "User ID")
        if not isinstance(memory_id, UUID):
            raise ValueError("Memory ID must be a valid UUID")

        return self.repository.get_by_id(
            memory_id=memory_id,
            user_id=user_id,
        )

    def get_session_memories(
        self,
        user_id: str,
        session_id: str,
    ) -> list[EpisodicMemory]:
        """
        Retrieves all episodic memories for a user within a session.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_non_empty_text(session_id, "Session ID")

        return self.repository.get_by_session(
            user_id=user_id,
            session_id=session_id,
        )

    def get_user_memories(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodicMemory]:
        """
        Retrieves paginated episodic memories belonging to a user.
        """
        self._validate_non_empty_text(user_id, "User ID")
        self._validate_min_int(limit, 1, "limit must be greater than zero")
        self._validate_min_int(offset, 0, "offset cannot be negative")

        return self.repository.get_by_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
