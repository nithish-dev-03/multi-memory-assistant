
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


class EmbeddingService:
    """
    Handles text embedding generation using Google's embedding model.
    """

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=settings.embedding_dimension,
        )

    def _validate_text(self, text: str) -> None:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

    def _validate_embedding(self, embedding: list[float]) -> list[float]:
        if len(embedding) != settings.embedding_dimension:
            raise ValueError(
                f"Expected {settings.embedding_dimension} dimensions, "
                f"received {len(embedding)}"
            )

        return embedding

    def embed_document(self, text: str) -> list[float]:
        """
        Generates an embedding for content that will be stored.

        Convert a memory into a vector before storing it

        RETRIEVAL_DOCUMENT
        """
        self._validate_text(text)

        embeddings = self.embedding_model.embed_documents(
            [text],
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=settings.embedding_dimension,
        )

        return self._validate_embedding(embeddings[0])

    def embed_query(self, text: str) -> list[float]:
        """
        Generates an embedding for a retrieval query.

        Convert a user's search query into a vector

        RETRIEVAL_QUERY
        """
        self._validate_text(text)

        embedding = self.embedding_model.embed_query(
            text,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=settings.embedding_dimension,
        )

        return self._validate_embedding(embedding)
