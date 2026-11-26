from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import app.utils.chat_services as chat_services


def _word_token_counter(_self, messages):
    """Deterministic local token counter based on word count."""
    return sum(len(str(msg.content).split()) for msg in messages)


@pytest.fixture
def token_counter_patch():
    with patch.object(
        type(chat_services.chat_summarize),
        "get_num_tokens_from_messages",
        _word_token_counter,
    ):
        yield


def test_history_below_token_limit_returns_original(token_counter_patch):
    system_msg = SystemMessage(content="You are a helpful assistant.")
    messages = [
        system_msg,
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]

    with patch.object(chat_services, "summarize_agent") as mock_summarize:
        result = chat_services.summarize_history(messages, max_tokens=100)

    assert result is messages
    mock_summarize.assert_not_called()


def test_history_exceeds_token_limit_summarizes_older_exchanges(token_counter_patch):
    system_msg = SystemMessage(content="System prompt.")
    ex1 = [
        HumanMessage(content=" ".join(["old_user"] * 10)),
        AIMessage(content=" ".join(["old_ai"] * 10)),
    ]
    ex2 = [HumanMessage(content="second user"), AIMessage(content="second ai")]
    ex3 = [HumanMessage(content="third user"), AIMessage(content="third ai")]
    messages = [system_msg, *ex1, *ex2, *ex3]

    with patch.object(
        chat_services,
        "summarize_agent",
        return_value="User shared old context.",
    ) as mock_summarize:
        result = chat_services.summarize_history(
            messages,
            max_tokens=20,
            recent_exchanges=2,
        )

    mock_summarize.assert_called_once_with(ex1)
    assert result[0] == system_msg
    assert isinstance(result[1], SystemMessage)
    assert result[1].content == "Conversation summary:\nUser shared old context."
    assert result[2:] == [*ex2, *ex3]


def test_previous_summary_is_included_in_new_summary(token_counter_patch):
    system_msg = SystemMessage(content="System prompt.")
    prev_summary = SystemMessage(content="Conversation summary:\nName is Nithish.")
    ex1 = [
        HumanMessage(content=" ".join(["backend"] * 12)),
        AIMessage(content=" ".join(["noted"] * 12)),
    ]
    ex2 = [HumanMessage(content="recent one"), AIMessage(content="reply one")]
    ex3 = [HumanMessage(content="recent two"), AIMessage(content="reply two")]
    messages = [system_msg, prev_summary, *ex1, *ex2, *ex3]

    with patch.object(
        chat_services,
        "summarize_agent",
        return_value="Nithish works on backend.",
    ) as mock_summarize:
        result = chat_services.summarize_history(
            messages,
            max_tokens=25,
            recent_exchanges=2,
        )

    mock_summarize.assert_called_once_with([prev_summary, *ex1])
    assert result[0] == system_msg
    assert result[1].content == "Conversation summary:\nNithish works on backend."
    assert result[2:] == [*ex2, *ex3]


def test_recent_exchanges_are_preserved_when_within_budget(token_counter_patch):
    system_msg = SystemMessage(content="System.")
    ex1 = [
        HumanMessage(content=" ".join(["archive"] * 15)),
        AIMessage(content=" ".join(["done"] * 15)),
    ]
    ex2 = [HumanMessage(content="keep two"), AIMessage(content="reply two")]
    ex3 = [HumanMessage(content="keep three"), AIMessage(content="reply three")]
    messages = [system_msg, *ex1, *ex2, *ex3]

    with patch.object(chat_services, "summarize_agent", return_value="Archived."):
        result = chat_services.summarize_history(
            messages,
            max_tokens=20,
            recent_exchanges=2,
        )

    assert result[2:] == [*ex2, *ex3]


def test_two_oversized_exchanges_summarize_oldest_and_keep_latest(token_counter_patch):
    system_msg = SystemMessage(content="System.")
    ex1 = [
        HumanMessage(content=" ".join(["verbose"] * 15)),
        AIMessage(content=" ".join(["answer"] * 15)),
    ]
    ex2 = [HumanMessage(content="latest question"), AIMessage(content="latest answer")]
    messages = [system_msg, *ex1, *ex2]

    with patch.object(
        chat_services,
        "summarize_agent",
        return_value="Condensed first exchange.",
    ) as mock_summarize:
        result = chat_services.summarize_history(
            messages,
            max_tokens=15,
            recent_exchanges=2,
        )

    mock_summarize.assert_called_once_with(ex1)
    assert result[0] == system_msg
    assert result[1].content == "Conversation summary:\nCondensed first exchange."
    assert result[2:] == ex2


def test_single_oversized_exchange_is_summarized(token_counter_patch):
    system_msg = SystemMessage(content="System.")
    ex1 = [
        HumanMessage(content=" ".join(["huge"] * 20)),
        AIMessage(content=" ".join(["response"] * 20)),
    ]
    messages = [system_msg, *ex1]

    with patch.object(
        chat_services,
        "summarize_agent",
        return_value="Compact summary.",
    ) as mock_summarize:
        result = chat_services.summarize_history(
            messages,
            max_tokens=10,
            recent_exchanges=2,
        )

    mock_summarize.assert_called_once_with(ex1)
    assert len(result) == 2
    assert result[0] == system_msg
    assert result[1].content == "Conversation summary:\nCompact summary."


def test_summary_exceeds_budget_raises_value_error(token_counter_patch):
    system_msg = SystemMessage(content="System prompt.")
    ex1 = [HumanMessage(content="hello there"), AIMessage(content="hi there")]
    messages = [system_msg, *ex1]

    with patch.object(
        chat_services,
        "summarize_agent",
        return_value=" ".join(["too_long"] * 20),
    ):
        with pytest.raises(ValueError, match="Summary exceeds token budget"):
            chat_services.summarize_history(
                messages,
                max_tokens=5,
                recent_exchanges=2,
            )
