from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.utils.prompt_budget import PromptBudgetManager, PromptBudgetResult


def _word_counter(messages):
    return sum(len(str(msg.content).split()) for msg in messages)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_prompt_tokens": 0}, "max_prompt_tokens must be greater than zero"),
        ({"max_prompt_tokens": True}, "max_prompt_tokens must be greater than zero"),
        ({"max_prompt_tokens": 100, "response_reserve_tokens": -1}, "response_reserve_tokens cannot be negative"),
        ({"max_prompt_tokens": 100, "response_reserve_tokens": 100}, "response_reserve_tokens must be less than max_prompt_tokens"),
        ({"max_prompt_tokens": 100, "response_reserve_tokens": 20, "memory_context_max_tokens": -1}, "memory_context_max_tokens cannot be negative"),
    ],
)
def test_prompt_budget_manager_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PromptBudgetManager(**kwargs)


def test_build_prompt_allocates_budget_and_keeps_user_message_once():
    manager = PromptBudgetManager(
        max_prompt_tokens=80,
        response_reserve_tokens=20,
        memory_context_max_tokens=30,
    )
    history = [
        SystemMessage(content="You are helpful."),
        SystemMessage(content="Conversation summary:\nUser builds APIs."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    original_snapshot = list(history)
    current_msg = HumanMessage(content="What stack do I use?")
    semantic = [(SimpleNamespace(content="Uses Python and FastAPI"), 0.1)]
    episodic = [SimpleNamespace(summary="Implemented pgvector schema")]

    result = manager.build_prompt(
        history_messages=history,
        current_user_message=current_msg,
        semantic_memories=semantic,
        episodic_memories=episodic,
        message_token_counter=_word_counter,
    )

    assert isinstance(result, PromptBudgetResult)
    assert history == original_snapshot
    assert result.total_input_tokens <= manager.available_input_tokens
    assert result.reserved_response_tokens == 20
    assert result.messages[-1] == current_msg
    assert sum(1 for m in result.messages if m == current_msg) == 1
    assert any("Relevant long-term facts:" in str(m.content) for m in result.messages)
    assert any("Conversation summary:" in str(m.content) for m in result.messages)


def test_build_prompt_handles_empty_memories_without_extra_system_message():
    manager = PromptBudgetManager(
        max_prompt_tokens=100,
        response_reserve_tokens=20,
        memory_context_max_tokens=30,
    )
    history = [SystemMessage(content="System")]
    current_msg = HumanMessage(content="Hi")

    result = manager.build_prompt(
        history_messages=history,
        current_user_message=current_msg,
        semantic_memories=[],
        episodic_memories=[],
        message_token_counter=_word_counter,
    )

    assert result.messages == [SystemMessage(content="System"), current_msg]
    assert result.memory_tokens == 0
    assert result.memory_context.formatted_context == ""


def test_build_prompt_trims_oldest_history_and_limits_memory_on_overflow():
    manager = PromptBudgetManager(
        max_prompt_tokens=30,
        response_reserve_tokens=10,
        memory_context_max_tokens=15,
    )
    # available_input_tokens = 20
    history = [
        SystemMessage(content="System"),
        HumanMessage(content="old question " * 6),
        AIMessage(content="old answer " * 6),
        HumanMessage(content="recent"),
        AIMessage(content="reply"),
    ]
    current_msg = HumanMessage(content="now")
    semantic = [(SimpleNamespace(content="Uses Python " * 10), 0.1)]

    result = manager.build_prompt(
        history_messages=history,
        current_user_message=current_msg,
        semantic_memories=semantic,
        episodic_memories=[],
        message_token_counter=_word_counter,
    )

    assert result.total_input_tokens <= 20
    assert len(history) == 5  # Persisted short-term history untouched.
    assert result.messages[-1] == current_msg
    assert all("old question" not in str(m.content) for m in result.messages)


def test_build_prompt_raises_when_base_messages_exceed_budget():
    manager = PromptBudgetManager(
        max_prompt_tokens=10,
        response_reserve_tokens=5,
        memory_context_max_tokens=2,
    )
    history = [SystemMessage(content="one two three four")]
    current_msg = HumanMessage(content="five six seven")

    with pytest.raises(ValueError, match="exceed available prompt budget"):
        manager.build_prompt(
            history_messages=history,
            current_user_message=current_msg,
            semantic_memories=[],
            episodic_memories=[],
            message_token_counter=_word_counter,
        )
