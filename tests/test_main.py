from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessageChunk

import app.main as main_module


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
