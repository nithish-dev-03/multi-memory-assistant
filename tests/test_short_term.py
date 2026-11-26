import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.memory.short_term import ShortTermMemory


def test_new_memory_returns_empty_list():
    memory = ShortTermMemory()

    assert memory.get_messages("session-1") == []


def test_add_message_stores_single_message():
    memory = ShortTermMemory()
    msg = HumanMessage(content="Hello")

    memory.add_message("session-1", msg)

    assert memory.get_messages("session-1") == [msg]


def test_add_messages_preserves_order():
    memory = ShortTermMemory()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hi"),
        AIMessage(content="Hello!"),
    ]

    memory.add_messages("session-1", messages)

    assert memory.get_messages("session-1") == messages


def test_get_last_n_messages_returns_correct_messages():
    memory = ShortTermMemory()
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="second"),
        HumanMessage(content="third"),
        AIMessage(content="fourth"),
    ]
    memory.add_messages("session-1", messages)

    assert memory.get_last_n_messages("session-1", 2) == messages[-2:]


def test_get_last_zero_messages_returns_empty_list():
    memory = ShortTermMemory()
    memory.add_message("session-1", HumanMessage(content="first"))

    assert memory.get_last_n_messages("session-1", 0) == []


def test_get_last_n_messages_negative_raises_value_error():
    memory = ShortTermMemory()

    with pytest.raises(ValueError, match="n cannot be negative"):
        memory.get_last_n_messages("session-1", -1)


def test_replace_messages_removes_old_history():
    memory = ShortTermMemory()
    memory.add_messages(
        "session-1",
        [
            HumanMessage(content="old user"),
            AIMessage(content="old ai"),
        ],
    )
    new_messages = [
        SystemMessage(content="Summary"),
        HumanMessage(content="new user"),
    ]

    memory.replace_messages("session-1", new_messages)

    assert memory.get_messages("session-1") == new_messages


def test_clear_removes_all_messages():
    memory = ShortTermMemory()
    memory.add_messages(
        "session-1",
        [
            HumanMessage(content="hello"),
            AIMessage(content="hi"),
        ],
    )

    memory.clear()

    assert memory.get_messages("session-1") == []


def test_add_message_to_session_a():
    memory = ShortTermMemory()
    msg_a = HumanMessage(content="Message for A")

    memory.add_message("session-a", msg_a)

    assert memory.get_messages("session-a") == [msg_a]


def test_add_message_to_session_b():
    memory = ShortTermMemory()
    msg_b = HumanMessage(content="Message for B")

    memory.add_message("session-b", msg_b)

    assert memory.get_messages("session-b") == [msg_b]


def test_session_isolation():
    memory = ShortTermMemory()
    msg_a = HumanMessage(content="Secret from A")
    msg_b = HumanMessage(content="Secret from B")

    memory.add_message("session-a", msg_a)
    memory.add_message("session-b", msg_b)

    assert memory.get_messages("session-a") == [msg_a]
    assert msg_b not in memory.get_messages("session-a")
    assert memory.get_messages("session-b") == [msg_b]
    assert msg_a not in memory.get_messages("session-b")


def test_unknown_session_returns_empty_list():
    memory = ShortTermMemory()
    memory.add_message("session-a", HumanMessage(content="Existing"))

    assert memory.get_messages("unknown-session") == []
    assert memory.get_last_n_messages("unknown-session", 2) == []


def test_retrieve_last_n_from_specific_session():
    memory = ShortTermMemory()
    msgs_a = [
        HumanMessage(content="A1"),
        AIMessage(content="A2"),
        HumanMessage(content="A3"),
    ]
    msgs_b = [
        HumanMessage(content="B1"),
        AIMessage(content="B2"),
    ]
    memory.add_messages("session-a", msgs_a)
    memory.add_messages("session-b", msgs_b)

    assert memory.get_last_n_messages("session-a", 2) == msgs_a[-2:]
    assert memory.get_last_n_messages("session-b", 1) == msgs_b[-1:]


def test_replace_session_history_only_affects_target_session():
    memory = ShortTermMemory()
    msgs_a = [HumanMessage(content="A old")]
    msgs_b = [HumanMessage(content="B keep")]
    new_a = [SystemMessage(content="A summary"), HumanMessage(content="A new")]

    memory.add_messages("session-a", msgs_a)
    memory.add_messages("session-b", msgs_b)

    memory.replace_messages("session-a", new_a)

    assert memory.get_messages("session-a") == new_a
    assert memory.get_messages("session-b") == msgs_b


def test_clear_one_session_leaves_others_untouched():
    memory = ShortTermMemory()
    msgs_a = [HumanMessage(content="A1")]
    msgs_b = [HumanMessage(content="B1")]
    memory.add_messages("session-a", msgs_a)
    memory.add_messages("session-b", msgs_b)

    memory.clear_session("session-a")

    assert memory.get_messages("session-a") == []
    assert memory.get_messages("session-b") == msgs_b


def test_clear_all_sessions_removes_all_sessions():
    memory = ShortTermMemory()
    memory.add_message("session-a", HumanMessage(content="A1"))
    memory.add_message("session-b", HumanMessage(content="B1"))

    memory.clear()

    assert memory.get_messages("session-a") == []
    assert memory.get_messages("session-b") == []
    assert memory.sessions == {}


@pytest.mark.parametrize("invalid_session_id", ["", "   ", "\t\n"])
def test_invalid_session_id_raises_value_error(invalid_session_id):
    memory = ShortTermMemory()
    msg = HumanMessage(content="Hello")

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.add_message(invalid_session_id, msg)

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.add_messages(invalid_session_id, [msg])

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.get_messages(invalid_session_id)

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.get_last_n_messages(invalid_session_id, 1)

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.replace_messages(invalid_session_id, [msg])

    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        memory.clear_session(invalid_session_id)

    assert memory.sessions == {}
