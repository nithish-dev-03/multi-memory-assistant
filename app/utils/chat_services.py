
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
)

from app.agents.summarize_agent import (
    chat_summarize,
    summarize_agent,
)


def _count_tokens(messages: list[BaseMessage]) -> int:
    """Count tokens in a list of LangChain messages."""
    token_count = chat_summarize.get_num_tokens_from_messages(messages)

    if type(token_count) is not int or token_count < 0:
        raise ValueError("Token count must be a non-negative integer")

    return token_count


def _build_history(
    system_message: SystemMessage | None,
    summary_message: SystemMessage,
    recent: list[list[BaseMessage]],
) -> list[BaseMessage]:
    """Build the updated conversation history."""
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
    Summarize older conversation exchanges while preserving recent ones.

    The resulting conversation history is validated against the token budget.
    """

    # Validate configuration.
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")

    if type(recent_exchanges) is not int or recent_exchanges < 0:
        raise ValueError("recent_exchanges cannot be negative")

    if current_tokens is not None:
        if type(current_tokens) is not int or current_tokens < 0:
            raise ValueError("current_tokens must be a non-negative integer")

    # Count the current conversation tokens.
    token_count = (
        current_tokens
        if current_tokens is not None
        else _count_tokens(messages)
    )

    # No summarization is required if history fits the budget.
    if token_count <= max_tokens:
        return messages

    # Preserve the original system message.
    system_message = (
        messages[0]
        if messages and isinstance(messages[0], SystemMessage)
        else None
    )

    conversation = messages[1:] if system_message else messages[:]

    # Extract an existing conversation summary, if available.
    previous_summary = None

    if conversation and isinstance(conversation[0], SystemMessage):
        previous_summary = conversation.pop(0)

    # Group messages into conversation exchanges.
    exchanges: list[list[BaseMessage]] = []
    current_exchange: list[BaseMessage] = []

    for message in conversation:
        current_exchange.append(message)

        if isinstance(message, AIMessage):
            exchanges.append(current_exchange)
            current_exchange = []

    # Preserve any incomplete exchange.
    if current_exchange:
        exchanges.append(current_exchange)

    # Separate older exchanges from recent exchanges.
    if recent_exchanges > 0:
        recent = exchanges[-recent_exchanges:]
        older = exchanges[:-recent_exchanges]
    else:
        recent = []
        older = exchanges[:]

    # Handle cases where there are no older exchanges to summarize.
    if not older and previous_summary is None:

        if not exchanges:
            raise ValueError(
                "History exceeds token budget with no exchanges to summarize. "
                f"Required: {token_count}, Available: {max_tokens}"
            )

        if len(exchanges) > 1:
            # Summarize the oldest exchange and preserve the remaining exchanges.
            older = exchanges[:1]
            recent = exchanges[1:]

        else:
            # Summarize the entire conversation if only one exchange exists.
            older = exchanges[:]
            recent = []

    # Prepare input for the summarization model.
    summary_input: list[BaseMessage] = []

    if previous_summary:
        summary_input.append(previous_summary)

    for exchange in older:
        summary_input.extend(exchange)

    # Generate an updated conversation summary.
    summary = summarize_agent(summary_input)

    summary_message = SystemMessage(
        content=f"Conversation summary:\n{summary}"
    )

    # Build the updated conversation history.
    updated_history = _build_history(
        system_message,
        summary_message,
        recent,
    )

    # Validate the resulting context against the token budget.
    while True:
        updated_tokens = _count_tokens(updated_history)

        if updated_tokens <= max_tokens:
            print(
                f"\nContext compressed: "
                f"{token_count} -> {updated_tokens} tokens"
            )

            return updated_history

        # Remove the oldest preserved exchange if the context is still too large.
        if recent:
            recent.pop(0)

            updated_history = _build_history(
                system_message,
                summary_message,
                recent,
            )

            continue

        # Even the summary cannot fit within the available budget.
        raise ValueError(
            "Summary exceeds token budget. "
            f"Required: {updated_tokens}, Available: {max_tokens}"
        )
