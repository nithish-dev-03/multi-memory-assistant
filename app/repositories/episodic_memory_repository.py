
from typing import Any
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.episodic_memory import EpisodicMemory


class EpisodicMemoryRepository:

    def __init__(self, db: Session):
        self.db = db

    @staticmethod
    def _validate_text(value: str, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def _validate_optional_text(
        value: str | None,
        field_name: str,
    ) -> None:
        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            raise ValueError(f"{field_name} cannot be empty")

    @staticmethod
    def _validate_min_int(
        value: int,
        min_value: int,
        message: str,
    ) -> None:
        if type(value) is not int or value < min_value:
            raise ValueError(message)

    def create(
        self,
        user_id: str,
        summary: str,
        session_id: str | None = None,
        event_type: str | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> EpisodicMemory:

        self._validate_text(user_id, "User ID")
        self._validate_text(summary, "Summary")

        self._validate_optional_text(session_id, "Session ID")
        self._validate_optional_text(event_type, "Event type")

        if metadata_json is not None and not isinstance(metadata_json, dict):
            raise ValueError("Metadata must be a dictionary")

        memory = EpisodicMemory(
            user_id=user_id,
            summary=summary,
            session_id=session_id,
            event_type=event_type,
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
    
    
    def get_by_id(
        self,
        memory_id: uuid.UUID,
        user_id: str,
    ) -> EpisodicMemory | None:

        self._validate_text(user_id, "User ID")

        if not isinstance(memory_id, uuid.UUID):
            raise ValueError("Memory ID must be a valid UUID")

        statement = select(EpisodicMemory).where(
            EpisodicMemory.id == memory_id,
            EpisodicMemory.user_id == user_id,
        )

        return self.db.execute(statement).scalar_one_or_none()

    
    def get_by_session(
        self,
        user_id: str,
        session_id: str,
    ) -> list[EpisodicMemory]:

        self._validate_text(user_id, "User ID")
        self._validate_text(session_id, "Session ID")

        statement = (
            select(EpisodicMemory)
            .where(
                EpisodicMemory.user_id == user_id,
                EpisodicMemory.session_id == session_id,
            )
            .order_by(
                EpisodicMemory.created_at.desc(),
                EpisodicMemory.id.desc(),
            )
        )

        return list(self.db.execute(statement).scalars().all())

    def get_by_user(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodicMemory]:

        self._validate_text(user_id, "User ID")
        self._validate_min_int(limit, 1, "limit must be greater than zero")
        self._validate_min_int(offset, 0, "offset cannot be negative")

        statement = (
            select(EpisodicMemory)
            .where(EpisodicMemory.user_id == user_id)
            .order_by(
                EpisodicMemory.created_at.desc(),
                EpisodicMemory.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        return list(self.db.execute(statement).scalars().all())

