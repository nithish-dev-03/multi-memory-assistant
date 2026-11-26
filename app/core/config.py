import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Holds environment variables, model, retrieval, and token-budget configuration.
    """

    EXPECTED_EMBEDDING_DIMENSION = 1536

    def __init__(self) -> None:
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
        self.sliding_window_size: int = int(
            os.getenv("SLIDING_WINDOW_SIZE") or os.getenv("SLIDING_WIN_SIZE") or 500
        )
        self.database_url: str | None = os.getenv("DATABASE_URL")
        self.embedding_dimension: int = int(
            os.getenv("EMBEDDING_DIMENSION", str(self.EXPECTED_EMBEDDING_DIMENSION))
        )
        if self.embedding_dimension != self.EXPECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"EMBEDDING_DIMENSION must be {self.EXPECTED_EMBEDDING_DIMENSION} "
                f"to match the database schema (got {self.embedding_dimension})"
            )

        # CLI development identity (never hardcoded to a production user).
        raw_cli_user_id = os.getenv("CLI_USER_ID")
        self.cli_user_id: str | None = (
            raw_cli_user_id.strip() if raw_cli_user_id and raw_cli_user_id.strip() else None
        )
        raw_cli_session_id = os.getenv("CLI_SESSION_ID", "cli-session")
        self.cli_session_id: str = raw_cli_session_id.strip() or "cli-session"

        # Unified prompt token budget configuration.
        self.max_prompt_tokens: int = int(os.getenv("MAX_PROMPT_TOKENS", "4000"))
        self.response_reserve_tokens: int = int(
            os.getenv("RESPONSE_RESERVE_TOKENS", "1000")
        )
        self.memory_context_max_tokens: int = int(
            os.getenv("MEMORY_CONTEXT_MAX_TOKENS", "1000")
        )
        if self.max_prompt_tokens <= 0:
            raise ValueError("MAX_PROMPT_TOKENS must be greater than zero")
        if (
            self.response_reserve_tokens < 0
            or self.response_reserve_tokens >= self.max_prompt_tokens
        ):
            raise ValueError(
                "RESPONSE_RESERVE_TOKENS must be >= 0 and < MAX_PROMPT_TOKENS"
            )
        if self.memory_context_max_tokens < 0:
            raise ValueError("MEMORY_CONTEXT_MAX_TOKENS cannot be negative")

        # Retrieval defaults.
        self.semantic_retrieval_limit: int = int(
            os.getenv("SEMANTIC_RETRIEVAL_LIMIT", "5")
        )
        self.episodic_retrieval_limit: int = int(
            os.getenv("EPISODIC_RETRIEVAL_LIMIT", "5")
        )
        raw_max_distance = os.getenv("SEMANTIC_MAX_DISTANCE")
        self.semantic_max_distance: float | None = (
            float(raw_max_distance)
            if raw_max_distance is not None and raw_max_distance.strip()
            else None
        )
        if self.semantic_retrieval_limit <= 0:
            raise ValueError("SEMANTIC_RETRIEVAL_LIMIT must be greater than zero")
        if self.episodic_retrieval_limit <= 0:
            raise ValueError("EPISODIC_RETRIEVAL_LIMIT must be greater than zero")
        if self.semantic_max_distance is not None and not (
            0.0 <= self.semantic_max_distance <= 2.0
        ):
            raise ValueError("SEMANTIC_MAX_DISTANCE must be between 0.0 and 2.0")


settings = Settings()
