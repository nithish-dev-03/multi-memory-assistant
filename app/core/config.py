import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    It holds all the environment variables and model configuration required for the application.
    """
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    sliding_window_size: int = int(
        os.getenv("SLIDING_WINDOW_SIZE") or os.getenv("SLIDING_WIN_SIZE") or 500
    )


settings = Settings()
