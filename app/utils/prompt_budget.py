from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.memory.context_builder import MemoryContext, MemoryContextBuilder


@dataclass(slots=True)
class PromptBudgetResult:
    """Assembled transient prompt messages and token budget accounting."""

    messages: list[BaseMessage]
    memory_context: MemoryContext
    total_input_tokens: int
    available_input_tokens: int
    reserved_response_tokens: int
    history_tokens: int
    memory_tokens: int


class PromptBudgetManager:
    """
    Coordinates unified prompt token budgeting across system instructions,
    retrieved long-term memory context, conversation summary, recent history,
    current user input, and reserved response capacity.
    """

    def __init__(
        self,
        max_prompt_tokens: int = 4000,
        response_reserve_tokens: int = 1000,
        memory_context_max_tokens: int = 1000,
    ) -> None:
        self._validate_int(max_prompt_tokens, 1, "max_prompt_tokens must be greater than zero")
        self._validate_int(
            response_reserve_tokens,
            0,
            "response_reserve_tokens cannot be negative",
        )
        if response_reserve_tokens >= max_prompt_tokens:
            raise ValueError(
                "response_reserve_tokens must be less than max_prompt_tokens"
            )
        self._validate_int(
            memory_context_max_tokens,
            0,
            "memory_context_max_tokens cannot be negative",
        )

        self.max_prompt_tokens = max_prompt_tokens
        self.response_reserve_tokens = response_reserve_tokens
        self.memory_context_max_tokens = memory_context_max_tokens
        self.available_input_tokens = max_prompt_tokens - response_reserve_tokens

    @staticmethod
    def _validate_int(value: int, min_value: int, message: str) -> None:
        if type(value) is not int or value < min_value:
            raise ValueError(message)

    @staticmethod
    def _count_messages_tokens(
        messages: list[BaseMessage],
        message_token_counter: Callable[[list[BaseMessage]], int],
    ) -> int:
        if not messages:
            return 0
        count = message_token_counter(messages)
        if type(count) is not int or count < 0:
            raise ValueError("Token count must be a non-negative integer")
        return count

    @staticmethod
    def _split_history(
        history_messages: list[BaseMessage],
    ) -> tuple[list[SystemMessage], list[SystemMessage], list[list[BaseMessage]]]:
        """Split persisted history into base system message, optional summary, and exchanges."""
        remaining = list(history_messages)
        system_prefix: list[SystemMessage] = []
        summary_prefix: list[SystemMessage] = []

        if remaining and isinstance(remaining[0], SystemMessage):
            system_prefix.append(remaining.pop(0))

        if remaining and isinstance(remaining[0], SystemMessage):
            summary_prefix.append(remaining.pop(0))

        exchanges: list[list[BaseMessage]] = []
        current_exchange: list[BaseMessage] = []
        for message in remaining:
            current_exchange.append(message)
            if isinstance(message, AIMessage):
                exchanges.append(current_exchange)
                current_exchange = []
        if current_exchange:
            exchanges.append(current_exchange)

        return system_prefix, summary_prefix, exchanges

    @staticmethod
    def _flatten_exchanges(exchanges: list[list[BaseMessage]]) -> list[BaseMessage]:
        return [message for exchange in exchanges for message in exchange]

    @staticmethod
    def _assemble_prompt(
        system_prefix: list[SystemMessage],
        memory_context: MemoryContext,
        summary_prefix: list[SystemMessage],
        exchanges: list[list[BaseMessage]],
        current_user_message: HumanMessage,
    ) -> list[BaseMessage]:
        prompt: list[BaseMessage] = list(system_prefix)
        if memory_context.formatted_context:
            prompt.append(
                SystemMessage(
                    content=(
                        "Relevant memory context (use only if helpful; do not fabricate):\n"
                        f"{memory_context.formatted_context}"
                    )
                )
            )
        prompt.extend(summary_prefix)
        prompt.extend(PromptBudgetManager._flatten_exchanges(exchanges))
        prompt.append(current_user_message)
        return prompt

    def build_prompt(
        self,
        history_messages: list[BaseMessage],
        current_user_message: HumanMessage,
        semantic_memories: list[tuple[Any, float]],
        episodic_memories: list[Any],
        message_token_counter: Callable[[list[BaseMessage]], int],
        text_token_counter: Callable[[str], int] | None = None,
    ) -> PromptBudgetResult:
        """
        Build a transient prompt within `available_input_tokens` without mutating
        persisted short-term history.
        """
        if not isinstance(current_user_message, HumanMessage):
            raise ValueError("current_user_message must be a HumanMessage")

        system_prefix, summary_prefix, exchanges = self._split_history(history_messages)

        base_tokens = self._count_messages_tokens(
            [*system_prefix, current_user_message],
            message_token_counter,
        )
        if base_tokens > self.available_input_tokens:
            raise ValueError(
                "System prompt and current user message exceed available prompt budget. "
                f"Required: {base_tokens}, Available: {self.available_input_tokens}"
            )

        # Trim transient copy of oldest exchanges (and summary as a last resort)
        # if history + current user message would exceed input capacity.
        working_summary = list(summary_prefix)
        working_exchanges = [list(exchange) for exchange in exchanges]

        while True:
            candidate_history_prompt = [
                *system_prefix,
                *working_summary,
                *self._flatten_exchanges(working_exchanges),
                current_user_message,
            ]
            history_plus_base_tokens = self._count_messages_tokens(
                candidate_history_prompt,
                message_token_counter,
            )
            if history_plus_base_tokens <= self.available_input_tokens:
                break
            if working_exchanges:
                working_exchanges.pop(0)
                continue
            if working_summary:
                working_summary.clear()
                continue
            raise ValueError(
                "Prompt exceeds available budget even after trimming history. "
                f"Required: {history_plus_base_tokens}, Available: {self.available_input_tokens}"
            )

        history_tokens = max(0, history_plus_base_tokens - base_tokens)
        remaining_for_memory = max(
            0,
            self.available_input_tokens - history_plus_base_tokens,
        )
        memory_budget = min(self.memory_context_max_tokens, remaining_for_memory)

        resolved_text_counter = text_token_counter or (
            lambda text: (
                0
                if not text
                else self._count_messages_tokens(
                    [SystemMessage(content=text)],
                    message_token_counter,
                )
            )
        )

        # Build memory context and account for wrapper message overhead if needed.
        while True:
            memory_context = MemoryContextBuilder(max_tokens=memory_budget).build(
                semantic_memories=semantic_memories,
                episodic_memories=episodic_memories,
                token_counter=resolved_text_counter,
            )
            prompt_messages = self._assemble_prompt(
                system_prefix=system_prefix,
                memory_context=memory_context,
                summary_prefix=working_summary,
                exchanges=working_exchanges,
                current_user_message=current_user_message,
            )
            total_input_tokens = self._count_messages_tokens(
                prompt_messages,
                message_token_counter,
            )

            if total_input_tokens <= self.available_input_tokens:
                break

            if memory_budget == 0 or not memory_context.formatted_context:
                raise ValueError(
                    "Final prompt exceeds available input token budget. "
                    f"Required: {total_input_tokens}, Available: {self.available_input_tokens}"
                )

            overshoot = total_input_tokens - self.available_input_tokens
            memory_budget = max(0, memory_budget - overshoot)

        return PromptBudgetResult(
            messages=prompt_messages,
            memory_context=memory_context,
            total_input_tokens=total_input_tokens,
            available_input_tokens=self.available_input_tokens,
            reserved_response_tokens=self.response_reserve_tokens,
            history_tokens=history_tokens,
            memory_tokens=memory_context.token_count,
        )
