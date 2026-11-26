import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    It holds all the environment variables and model configuration required for the application.
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


settings = Settings()
