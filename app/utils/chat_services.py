
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from app.agents.sumarize_agent import (
    chat_summarize,
    sumarize_agent,
)


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

    recent = exchanges[-recent_exchanges:]

    older = exchanges[:-recent_exchanges]

    if not older and previous_summary is None:
        return messages

    summary_input = []

    if previous_summary:
        summary_input.append(previous_summary)

    for exchange in older:
        summary_input.extend(exchange)

    summary = sumarize_agent(summary_input)

    summary_message = SystemMessage(
        content=f"Conversation summary:\n{summary}"
    )

    recent_messages = [
        message
        for exchange in recent
        for message in exchange
    ]

    updated_history = []

    if system_message:
        updated_history.append(system_message)

    updated_history.append(summary_message)
    updated_history.extend(recent_messages)

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

            recent_messages = [
                message
                for exchange in recent
                for message in exchange
            ]

            updated_history = []

            if system_message:
                updated_history.append(system_message)

            updated_history.append(summary_message)
            updated_history.extend(recent_messages)

            continue

        raise ValueError(
            f"Summary exceeds token budget. "
            f"Required: {updated_tokens}, "
            f"Available: {max_tokens}"
        )
