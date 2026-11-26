from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.memory.context_builder import MemoryContextBuilder


def test_init_accepts_non_negative_max_tokens():
    builder = MemoryContextBuilder(max_tokens=0)
    assert builder.max_tokens == 0


@pytest.mark.parametrize("invalid_max_tokens", [-1, -100, True, 10.5])
def test_init_rejects_invalid_max_tokens(invalid_max_tokens):
    with pytest.raises(ValueError, match="max_tokens cannot be negative"):
        MemoryContextBuilder(max_tokens=invalid_max_tokens)


def test_build_creates_formatted_context_within_token_budget():
    builder = MemoryContextBuilder(max_tokens=20)
    semantic = [
        (SimpleNamespace(content="Uses Python"), 0.1),
        (SimpleNamespace(content="Uses FastAPI"), 0.2),
    ]
    episodic = [
        SimpleNamespace(summary="Built a retrieval service")
    ]

    result = builder.build(
        semantic_memories=semantic,
        episodic_memories=episodic,
        token_counter=lambda text: len(text.split()),
    )

    assert isinstance(result.formatted_context, str)
    assert "Relevant long-term facts:" in result.formatted_context
    assert "Recent experiences:" in result.formatted_context
    assert result.semantic_memories == semantic
    assert result.episodic_memories == episodic
    assert result.token_count == len(result.formatted_context.split())
    assert result.token_count <= builder.max_tokens


def test_build_prioritizes_deduplicates_and_skips_oversized_memories():
    builder = MemoryContextBuilder(max_tokens=16)
    close_sem = (SimpleNamespace(content="Uses Python"), 0.1)
    dup_sem = (SimpleNamespace(content="uses   python"), 0.3)
    huge_sem = (SimpleNamespace(content="one two three four five six seven eight nine ten"), 0.2)
    small_sem = (SimpleNamespace(content="Uses SQL"), 0.4)

    older_ep = SimpleNamespace(
        summary="Created DB models",
        created_at=datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc),
    )
    newer_ep = SimpleNamespace(
        summary="Built retrieval",
        created_at=datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc),
    )

    result = builder.build(
        semantic_memories=[small_sem, dup_sem, huge_sem, close_sem],
        episodic_memories=[older_ep, newer_ep],
        token_counter=lambda text: len(text.split()),
    )

    assert result.semantic_memories == [close_sem, small_sem]
    assert result.episodic_memories == [newer_ep]
    assert result.token_count <= builder.max_tokens
    assert "Built retrieval" in result.formatted_context
    assert "Created DB models" not in result.formatted_context


def test_extract_memory_text_rejects_non_string_and_empty_tuple():
    assert MemoryContextBuilder._extract_memory_text(()) is None
    assert MemoryContextBuilder._extract_memory_text(SimpleNamespace(content=123)) is None
    assert MemoryContextBuilder._extract_memory_text(SimpleNamespace(summary=["bad"])) is None


def test_build_deduplicates_across_semantic_and_episodic_stores():
    builder = MemoryContextBuilder(max_tokens=50)
    sem = [(SimpleNamespace(content="User built FastAPI APIs"), 0.1)]
    ep = [
        SimpleNamespace(summary="  user built  fastapi apis "),
        SimpleNamespace(summary="Configured Alembic migrations"),
    ]

    result = builder.build(
        semantic_memories=sem,
        episodic_memories=ep,
        token_counter=lambda text: len(text.split()),
    )

    assert result.semantic_memories == sem
    assert result.episodic_memories == [ep[1]]
    assert result.formatted_context.count("FastAPI") == 1


def test_prioritize_episodic_memories_orders_newest_first_and_none_last():
    oldest = SimpleNamespace(
        id="oldest",
        created_at=datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc),
    )
    newest = SimpleNamespace(
        id="newest",
        created_at=datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc),
    )
    missing_a = SimpleNamespace(id="missing_a", created_at=None)
    missing_b = SimpleNamespace(id="missing_b")

    result = MemoryContextBuilder._prioritize_episodic_memories(
        [missing_a, oldest, missing_b, newest]
    )

    assert [item.id for item in result] == [
        "newest",
        "oldest",
        "missing_a",
        "missing_b",
    ]


def test_count_tokens_returns_valid_count():
    count = MemoryContextBuilder._count_tokens("hello world", lambda text: len(text.split()))
    assert count == 2


@pytest.mark.parametrize("invalid_count", [-1, -5, True, 1.5])
def test_count_tokens_rejects_negative_or_invalid_count(invalid_count):
    with pytest.raises(ValueError, match="Token count cannot be negative"):
        MemoryContextBuilder._count_tokens("hello", lambda _text: invalid_count)


def test_fit_memories_within_budget_rejects_negative_token_count():
    builder = MemoryContextBuilder(max_tokens=100)
    memories = [SimpleNamespace(content="hello")]

    with pytest.raises(ValueError, match="Token count cannot be negative"):
        builder._fit_memories_within_budget(memories, 50, lambda _text: -1)


def test_prioritize_semantic_memories_sorts_by_ascending_cosine_distance():
    far = (SimpleNamespace(content="Far memory"), 0.75)
    close = (SimpleNamespace(content="Close memory"), 0.12)
    medium = (SimpleNamespace(content="Medium memory"), 0.35)

    result = MemoryContextBuilder._prioritize_semantic_memories([far, close, medium])

    assert result == [close, medium, far]


def test_deduplicate_memories_supports_semantic_tuples_and_episodic_summaries():
    builder = MemoryContextBuilder()
    sem_1 = (SimpleNamespace(content="  Uses   Python "), 0.1)
    sem_dup = (SimpleNamespace(content="uses python"), 0.4)
    sem_empty = (SimpleNamespace(content="   "), 0.2)
    ep_1 = SimpleNamespace(summary="Built FastAPI routes.")
    ep_dup = SimpleNamespace(summary="built  fastapi   routes.")
    ep_2 = SimpleNamespace(summary="Added pgvector index.")

    assert builder._deduplicate_memories([sem_1, sem_empty, sem_dup]) == [sem_1]
    assert builder._deduplicate_memories([ep_1, ep_dup, ep_2]) == [ep_1, ep_2]


def test_fit_memories_within_budget_supports_semantic_tuples_and_episodic_summaries():
    builder = MemoryContextBuilder(max_tokens=100)
    sem_fit = (SimpleNamespace(content="one two"), 0.1)
    sem_too_big = (SimpleNamespace(content="one two three four five"), 0.2)
    sem_also_fits = (SimpleNamespace(content="three"), 0.3)

    selected_sem, used_sem = builder._fit_memories_within_budget(
        [sem_fit, sem_too_big, sem_also_fits],
        remaining_tokens=3,
        token_counter=lambda text: len(text.split()),
    )
    assert selected_sem == [sem_fit, sem_also_fits]
    assert used_sem == 3

    ep_fit = SimpleNamespace(summary="short summary")
    ep_too_big = SimpleNamespace(summary="this summary is way too long")
    selected_ep, used_ep = builder._fit_memories_within_budget(
        [ep_fit, ep_too_big],
        remaining_tokens=2,
        token_counter=lambda text: len(text.split()),
    )
    assert selected_ep == [ep_fit]
    assert used_ep == 2
