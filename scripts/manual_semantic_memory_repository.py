import uuid

from app.core.database import SessionLocal
from app.repositories.semantic_memory_repository import (
    SemanticMemoryRepository,
)
from app.services.embedding_service import EmbeddingService


def main():
    embedding_service = EmbeddingService()
    db = SessionLocal()

    try:
        repository = SemanticMemoryRepository(db)

        query = "Which frontend technologies does this developer use?"
        query_embedding = embedding_service.embed_query(query)

        similar_memories = repository.search_similar(
            user_id="test-user-001",
            query_embedding=query_embedding,
            top_k=5,
        )
        print(f"\nFound {len(similar_memories)} similar memories:\n")
        for index, memory in enumerate(similar_memories, start=1):
            print(f"{index}. {memory.content}")
            print(f"   Category: {memory.category}")
            print(f"   User: {memory.user_id}")

        memories = repository.get_by_user(
            user_id="test-user-001",
            limit=10,
        )
        assert all(
            memory.user_id == "test-user-001"
            for memory in memories
        )
        print("User memories:", len(memories))

        other_memories = repository.get_by_user(
            user_id="other-user-999",
        )
        assert len(other_memories) == 0
        print("User isolation verified!")

        memory_id = uuid.UUID("c886a39a-960f-4ee4-8f34-c2d9914d2d53")
        found = repository.get_by_id(memory_id=memory_id, user_id="test-user-001")
        missing = repository.get_by_id(memory_id=memory_id, user_id="other-user-999")
        assert found is not None and found.id == memory_id
        assert missing is None
        print("get_by_id user scoping verified!")

    finally:
        db.close()


if __name__ == "__main__":
    main()
