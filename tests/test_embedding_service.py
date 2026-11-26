
from app.services.embedding_service import EmbeddingService


def main():
    service = EmbeddingService()

    document = "I am a software developer working with React and Python."

    document_embedding = service.embed_document(document)
    query_embedding = service.embed_query(
        "What programming technologies do I work with?"
    )

    print("Document dimensions:", len(document_embedding))
    print("Query dimensions:", len(query_embedding))

    assert len(document_embedding) == 1536
    assert len(query_embedding) == 1536

    print("Embedding service working successfully!")


if __name__ == "__main__":
    main()
