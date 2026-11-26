from langchain_core.messages import BaseMessage
class ShortTermMemory:

    def __init__(self):
        self.sessions: dict[str, list[BaseMessage]] = {}

    def _validate_session_id(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("Session ID cannot be empty")

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        self._validate_session_id(session_id)
        self.sessions.setdefault(session_id, []).append(message)

    def add_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        self._validate_session_id(session_id)
        self.sessions.setdefault(session_id, []).extend(messages)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        self._validate_session_id(session_id)
        return self.sessions.get(session_id, []).copy()

    def get_last_n_messages(self, session_id: str, n: int) -> list[BaseMessage]:
        self._validate_session_id(session_id)

        if n < 0:
            raise ValueError("n cannot be negative")

        if n == 0:
            return []

        return self.get_messages(session_id)[-n:]

    def replace_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        self._validate_session_id(session_id)
        self.sessions[session_id] = messages.copy()

    def clear_session(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        self.sessions.pop(session_id, None)

    def clear(self) -> None:
        self.sessions.clear()
