from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

import app.main as main_module
from app.memory.short_term import ShortTermMemory


def test_create_convo_rolls_back_user_message_on_stream_error():
    seen_histories = []

    def fake_stream(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        if len(seen_histories) == 1:
            raise RuntimeError("stream failed")
        return [AIMessageChunk(content="recovered")]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    fake_model.get_num_tokens_from_messages.return_value = 10

    inputs = iter(["first attempt", "second attempt", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    assert seen_histories == [
        [("system", "You are a helpful assistant."), ("human", "first attempt")],
        [("system", "You are a helpful assistant."), ("human", "second attempt")],
    ]


def test_create_convo_rolls_back_full_exchange_on_summarization_error():
    seen_histories = []

    def fake_stream(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        return [AIMessageChunk(content="assistant reply")]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    fake_model.get_num_tokens_from_messages.return_value = 9999

    inputs = iter(["turn one", "turn two", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(
            main_module,
            "summarize_history",
            side_effect=ValueError("summary failed"),
        ),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    assert seen_histories == [
        [("system", "You are a helpful assistant."), ("human", "turn one")],
        [("system", "You are a helpful assistant."), ("human", "turn two")],
    ]


def test_create_convo_persists_history_in_short_term_memory():
    memory = ShortTermMemory()
    seen_histories = []
    replies = iter(["hi there", "doing great"])

    def fake_stream(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        return [AIMessageChunk(content=next(replies))]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    fake_model.get_num_tokens_from_messages.return_value = 10

    inputs = iter(["hello", "how are you?", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(main_module, "ShortTermMemory", return_value=memory),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    assert memory.get_messages("cli-session") == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="how are you?"),
        AIMessage(content="doing great"),
    ]
    assert seen_histories[1] == [
        ("system", "You are a helpful assistant."),
        ("human", "hello"),
        ("ai", "hi there"),
        ("human", "how are you?"),
    ]


def test_create_convo_replaces_session_history_after_summarization():
    memory = ShortTermMemory()
    seen_histories = []
    replies = iter(["hello back", "follow up reply"])

    def fake_stream(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        return [AIMessageChunk(content=next(replies))]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    # Turn 1 check (>=50) -> final check (20) -> Turn 2 check (<50) -> final check (30)
    fake_model.get_num_tokens_from_messages.side_effect = [60, 20, 30, 35]

    compressed_history = [
        SystemMessage(content="You are a helpful assistant."),
        SystemMessage(content="Conversation summary:\nUser greeted assistant."),
    ]

    inputs = iter(["hello", "follow up", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(main_module, "ShortTermMemory", return_value=memory),
        patch.object(main_module.settings, "sliding_window_size", 50),
        patch.object(
            main_module,
            "summarize_history",
            return_value=compressed_history,
        ) as mock_summarize,
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    mock_summarize.assert_called_once()
    assert seen_histories[1] == [
        ("system", "You are a helpful assistant."),
        ("system", "Conversation summary:\nUser greeted assistant."),
        ("human", "follow up"),
    ]
    assert memory.get_messages("cli-session") == [
        *compressed_history,
        HumanMessage(content="follow up"),
        AIMessage(content="follow up reply"),
    ]


def test_create_convo_rolls_back_after_partial_stream_failure():
    memory = ShortTermMemory()
    seen_histories = []

    def partial_then_fail(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        if len(seen_histories) == 1:
            def failing_generator():
                yield AIMessageChunk(content="partial ")
                raise RuntimeError("stream interrupted")

            return failing_generator()
        return [AIMessageChunk(content="complete reply")]

    fake_model = MagicMock()
    fake_model.stream.side_effect = partial_then_fail
    fake_model.get_num_tokens_from_messages.return_value = 10

    inputs = iter(["interrupted turn", "clean turn", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(main_module, "ShortTermMemory", return_value=memory),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    assert seen_histories == [
        [("system", "You are a helpful assistant."), ("human", "interrupted turn")],
        [("system", "You are a helpful assistant."), ("human", "clean turn")],
    ]
    assert memory.get_messages("cli-session") == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="clean turn"),
        AIMessage(content="complete reply"),
    ]
