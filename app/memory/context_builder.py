
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


@dataclass
class MemoryContext:
    """
    Represents the final memory context prepared for the LLM.
    """

    semantic_memories: list[Any] = field(default_factory=list)
    episodic_memories: list[Any] = field(default_factory=list)
    formatted_context: str = ""
    token_count: int = 0


class MemoryContextBuilder:
    """
    Prepares retrieved memories for injection into the LLM context.
    """

    def __init__(self, max_tokens: int = 1500):
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")
        self.max_tokens = max_tokens

    def build(
        self,
        semantic_memories: list[Any],
        episodic_memories: list[Any],
    ) -> MemoryContext:
        """
        Builds a structured context from semantic and episodic memories.
        """

        # We will implement filtering, prioritization,
        # deduplication and token budgeting here.
        raise NotImplementedError
    

    @staticmethod
    def _extract_memory_text(item: Any) -> str | None:
        """
        Extract text from a semantic `(memory, distance)` tuple, a semantic
        memory (`content`), or an episodic memory (`summary`).
        """
        memory = item[0] if isinstance(item, tuple) else item
        text = getattr(memory, "content", None)
        if text is None:
            text = getattr(memory, "summary", None)
        return text

    @staticmethod
    def _normalize_content(content: str) -> str:
        """
        Normalize memory content for duplicate comparison.
        """
        return " ".join(content.lower().split())

    def _deduplicate_memories(
        self,
        memories: list[Any],
    ) -> list[Any]:
        """
        Remove memories containing duplicate content.
        Preserve the original order.
        """
        unique_memories = []
        seen_contents = set()

        for memory in memories:
            content = self._extract_memory_text(memory)

            if not content or not content.strip():
                continue

            normalized_content = self._normalize_content(content)

            if normalized_content in seen_contents:
                continue

            seen_contents.add(normalized_content)
            unique_memories.append(memory)

        return unique_memories


    @staticmethod
    def _prioritize_semantic_memories(
        memories: list[tuple[Any, float]],
    ) -> list[tuple[Any, float]]:
        """Sort semantic matches by ascending cosine distance."""
        return sorted(memories, key=lambda item: item[1])

    @staticmethod
    def _prioritize_episodic_memories(
        memories: list[Any],
    ) -> list[Any]:
        """
        Sort episodic memories by creation time.
        Memories without a creation time come last.
        """
        return sorted(
            memories,
            key=lambda memory: (
                getattr(memory, "created_at", None) is not None,
                getattr(memory, "created_at", None),
            ),
            reverse=True,
        )


    @staticmethod
    def _count_tokens(
        text: str,
        token_counter: Callable[[str], int],
    ) -> int:
        """
        Count tokens using the application's configured tokenizer.
        """
        token_count = token_counter(text)
        if type(token_count) is not int or token_count < 0:
            raise ValueError("Token count cannot be negative")
        return token_count

    def _fit_memories_within_budget(
        self,
        memories: list[Any],
        remaining_tokens: int,
        token_counter: Callable[[str], int],
    ) -> tuple[list[Any], int]:
        """
        Select memories that fit within the available token budget.
        Preserves the input priority order.
        """
        selected_memories = []
        used_tokens = 0

        for memory in memories:
            content = self._extract_memory_text(memory)

            if not content or not content.strip():
                continue

            memory_tokens = self._count_tokens(
                content,
                token_counter,
            )

            if used_tokens + memory_tokens > remaining_tokens:
                continue

            selected_memories.append(memory)
            used_tokens += memory_tokens

        return selected_memories, used_tokens

