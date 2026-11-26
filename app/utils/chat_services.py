
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
)

from app.agents.summarize_agent import (
    chat_summarize,
    summarize_agent,
)


def _build_history(
    system_message: SystemMessage | None,
    summary_message: SystemMessage,
    recent: list[list[BaseMessage]],
) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    if system_message:
        history.append(system_message)
    history.append(summary_message)
    history.extend(
        message
        for exchange in recent
        for message in exchange
    )
    return history


def summarize_history(
    messages: list[BaseMessage],
    max_tokens: int,
    current_tokens: int | None = None,
    recent_exchanges: int = 2,
) -> list[BaseMessage]:
    """
    Summarize older exchanges while preserving recent exchanges.

    Also validates the resulting context token count.
    """

    token_count = (
        current_tokens
        if current_tokens is not None
        else chat_summarize.get_num_tokens_from_messages(messages)
    )

    if token_count < max_tokens:
        return messages

    system_message = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else None
    )

    conversation = messages[1:] if system_message else messages[:]

    previous_summary = None

    if conversation and isinstance(conversation[0], SystemMessage):
        previous_summary = conversation.pop(0)

    exchanges = []
    current_exchange = []

    for message in conversation:
        current_exchange.append(message)

        if isinstance(message, AIMessage):
            exchanges.append(current_exchange)
            current_exchange = []

    if current_exchange:
        exchanges.append(current_exchange)

    if recent_exchanges > 0:
        recent = exchanges[-recent_exchanges:]
        older = exchanges[:-recent_exchanges]
    else:
        recent = []
        older = exchanges[:]

    if not older and previous_summary is None:
        if not exchanges:
            raise ValueError(
                f"History exceeds token budget with no exchanges to summarize. "
                f"Required: {token_count}, Available: {max_tokens}"
            )
        if len(exchanges) > 1:
            # Fallback: summarize the oldest exchange while retaining remaining recent context.
            older = exchanges[:1]
            recent = exchanges[1:]
        else:
            # Single oversized exchange: summarize it completely to fit within budget.
            older = exchanges[:]
            recent = []

    summary_input = []

    if previous_summary:
        summary_input.append(previous_summary)

    for exchange in older:
        summary_input.extend(exchange)

    summary = summarize_agent(summary_input)

    summary_message = SystemMessage(
        content=f"Conversation summary:\n{summary}"
    )

    updated_history = _build_history(
        system_message,
        summary_message,
        recent,
    )

    while True:
        updated_tokens = (
            chat_summarize.get_num_tokens_from_messages(
                updated_history
            )
        )

        if updated_tokens <= max_tokens:
            print(
                f"\nContext compressed: "
                f"{token_count} -> {updated_tokens} tokens"
            )
            return updated_history

        if len(recent) > 0:
            recent.pop(0)
            updated_history = _build_history(
                system_message,
                summary_message,
                recent,
            )
            continue

        raise ValueError(
            f"Summary exceeds token budget. "
            f"Required: {updated_tokens}, "
            f"Available: {max_tokens}"
        )
