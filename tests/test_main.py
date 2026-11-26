from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

import app.main as main_module
from app.core.database import get_db_session
from app.memory.retrieval import MemoryRetrievalResult
from app.memory.short_term import ShortTermMemory


@contextmanager
def _noop_db_session():
    yield MagicMock()


def _passthrough_prompt_budget(
    history_messages,
    current_user_message,
    semantic_memories,
    episodic_memories,
    message_token_counter,
    text_token_counter=None,
):
    return SimpleNamespace(
        messages=[*history_messages, current_user_message],
        total_input_tokens=0,
    )


@pytest.fixture
def isolate_cli_external_services():
    fake_retrieval = MagicMock()
    fake_retrieval.retrieve_memories.return_value = MemoryRetrievalResult(
        semantic_memories=[],
        episodic_memories=[],
    )
    fake_agent = MagicMock()
    fake_budget = MagicMock()
    fake_budget.build_prompt.side_effect = _passthrough_prompt_budget

    with (
        patch.object(main_module.settings, "cli_user_id", "test-user"),
        patch.object(main_module, "get_db_session", _noop_db_session),
        patch.object(main_module, "MemoryRetrievalService", return_value=fake_retrieval),
        patch.object(main_module, "MemoryManagerAgent", return_value=fake_agent),
        patch.object(main_module, "PromptBudgetManager", return_value=fake_budget),
        patch.object(main_module, "SemanticMemoryService"),
        patch.object(main_module, "EpisodicMemoryService"),
    ):
        yield


def test_create_convo_rolls_back_user_message_on_stream_error(isolate_cli_external_services):
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


def test_create_convo_rolls_back_full_exchange_on_summarization_error(
    isolate_cli_external_services,
):
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


def test_create_convo_persists_history_in_short_term_memory(
    isolate_cli_external_services,
):
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


def test_create_convo_replaces_session_history_after_summarization(
    isolate_cli_external_services,
):
    memory = ShortTermMemory()
    seen_histories = []
    replies = iter(["hello back", "follow up reply"])

    def fake_stream(history):
        seen_histories.append([(msg.type, msg.content) for msg in history])
        return [AIMessageChunk(content=next(replies))]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    # Turn 1 check (>50) -> final check (20) -> Turn 2 check (<50) -> final check (30)
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


def test_create_convo_rolls_back_after_partial_stream_failure(
    isolate_cli_external_services,
):
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


def test_create_convo_exit_immediately_does_not_raise_unbound_error(
    isolate_cli_external_services,
):
    fake_model = MagicMock()
    inputs = iter(["exit"])

    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo()

    fake_model.stream.assert_not_called()


def test_create_convo_injects_retrieved_memories_without_persisting_them():
    memory = ShortTermMemory()
    seen_prompts = []

    def fake_stream(messages):
        seen_prompts.append(list(messages))
        return [AIMessageChunk(content="You use Python and FastAPI.")]

    fake_model = MagicMock()
    fake_model.stream.side_effect = fake_stream
    fake_model.get_num_tokens_from_messages.side_effect = (
        lambda msgs: sum(len(str(m.content).split()) for m in msgs)
    )

    fake_retrieval = MagicMock()
    fake_retrieval.retrieve_memories.return_value = MemoryRetrievalResult(
        semantic_memories=[(SimpleNamespace(content="User codes in Python"), 0.1)],
        episodic_memories=[SimpleNamespace(summary="Built FastAPI routes yesterday")],
    )
    fake_agent = MagicMock()
    closed_sessions = []

    @contextmanager
    def tracking_session():
        db = MagicMock()
        try:
            yield db
        finally:
            closed_sessions.append(db)

    inputs = iter(["What stack do I use?", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(main_module, "ShortTermMemory", return_value=memory),
        patch.object(main_module, "get_db_session", tracking_session),
        patch.object(main_module, "MemoryRetrievalService", return_value=fake_retrieval),
        patch.object(main_module, "MemoryManagerAgent", return_value=fake_agent),
        patch.object(main_module, "SemanticMemoryService"),
        patch.object(main_module, "EpisodicMemoryService"),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo(user_id="dev-user-1", session_id="sess-99")

    assert len(closed_sessions) == 2
    prompt_sent = seen_prompts[0]
    assert any("User codes in Python" in str(m.content) for m in prompt_sent)
    assert any("Built FastAPI routes yesterday" in str(m.content) for m in prompt_sent)
    assert sum(1 for m in prompt_sent if m == HumanMessage(content="What stack do I use?")) == 1

    # Persisted short-term history must NOT contain injected retrieval context.
    stored = memory.get_messages("sess-99")
    assert stored == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What stack do I use?"),
        AIMessage(content="You use Python and FastAPI."),
    ]
    fake_agent.process_turn.assert_called_once()


def test_create_convo_memory_write_failure_does_not_corrupt_history():
    memory = ShortTermMemory()
    fake_model = MagicMock()
    fake_model.stream.return_value = [AIMessageChunk(content="Stored cleanly.")]
    fake_model.get_num_tokens_from_messages.side_effect = (
        lambda msgs: sum(len(str(m.content).split()) for m in msgs)
    )

    fake_retrieval = MagicMock()
    fake_retrieval.retrieve_memories.return_value = MemoryRetrievalResult(
        semantic_memories=[],
        episodic_memories=[],
    )
    fake_agent = MagicMock()
    fake_agent.process_turn.side_effect = RuntimeError("memory write failed")

    inputs = iter(["Remember I like Rust", "exit"])
    with (
        patch.object(main_module, "create_chat_model", return_value=fake_model),
        patch.object(main_module, "ShortTermMemory", return_value=memory),
        patch.object(main_module, "get_db_session", _noop_db_session),
        patch.object(main_module, "MemoryRetrievalService", return_value=fake_retrieval),
        patch.object(main_module, "MemoryManagerAgent", return_value=fake_agent),
        patch.object(main_module, "SemanticMemoryService"),
        patch.object(main_module, "EpisodicMemoryService"),
        patch("builtins.input", side_effect=lambda _: next(inputs)),
    ):
        main_module.create_convo(user_id="dev-user-1", session_id="sess-99")

    assert memory.get_messages("sess-99") == [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Remember I like Rust"),
        AIMessage(content="Stored cleanly."),
    ]


def test_get_db_session_rolls_back_and_closes_on_error():
    mock_session = MagicMock()
    with patch("app.core.database.SessionLocal", return_value=mock_session):
        with get_db_session() as db:
            assert db is mock_session
        mock_session.commit.assert_not_called()
        mock_session.close.assert_called_once()

    mock_session_err = MagicMock()
    with patch("app.core.database.SessionLocal", return_value=mock_session_err):
        with pytest.raises(RuntimeError, match="boom"):
            with get_db_session():
                raise RuntimeError("boom")
        mock_session_err.rollback.assert_called_once()
        mock_session_err.close.assert_called_once()
