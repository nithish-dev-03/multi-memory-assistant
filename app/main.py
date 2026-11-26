
from app.agents.chat_agent import create_chat_model
from app.memory.short_term import ShortTermMemory
from app.utils.chat_services import summarize_history
from app.core.config import settings

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    AIMessage,
)


def count_token(model, messages: list[BaseMessage]) -> int:
    """Count tokens in the supplied messages."""
    return model.get_num_tokens_from_messages(messages)


def create_convo():
    model = create_chat_model(settings.chat_model)

    memory = ShortTermMemory()
    session_id = "cli-session"

    # Initialize the session with a system message.
    if not memory.get_messages(session_id):
        memory.add_message(
            session_id,
            SystemMessage(content="You are a helpful assistant.")
        )

    while True:
        text = input("\nYou: ").strip()

        if text.lower() in {"exit", "quit", "bye"}:
            print("Assistant: Goodbye!")
            break

        if not text:
            continue

        # 1. Backup existing history for rollback.
        previous_history = memory.get_messages(session_id)

        # 2. Add the user's message.
        memory.add_message(
            session_id,
            HumanMessage(content=text)
        )

        # 3. Retrieve updated history.
        message_history = memory.get_messages(session_id)

        print("\nAssistant: ", end="", flush=True)
        ai_response = ""

        try:
            # 4. Generate the assistant response.
            for chunk in model.stream(message_history):
                content = chunk.content

                if isinstance(content, str):
                    print(content, end="", flush=True)
                    ai_response += content

                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text_content = block.get("text", "")

                            if text_content:
                                print(text_content, end="", flush=True)
                                ai_response += text_content

            print("\n")

            # 5. Store the assistant response.
            memory.add_message(
                session_id,
                AIMessage(content=ai_response)
            )

            # 6. Retrieve complete history and count tokens.
            message_history = memory.get_messages(session_id)

            tokens = count_token(model, message_history)
            print(f"Tokens: {tokens}")

            max_tokens = settings.sliding_window_size

            # 7. Summarize if the token threshold is reached.
            if tokens >= max_tokens:
                message_history = summarize_history(
                    message_history,
                    max_tokens,
                    current_tokens=tokens
                )

                # 8. Save compressed history.
                memory.replace_messages(
                    session_id,
                    message_history
                )

            # Display final stored history.
            print("\n--- STORED HISTORY ---")

            for index, message in enumerate(message_history):
                print(
                    f"{index}: "
                    f"{message.type} -> {message.content}"
                )

            final_tokens = count_token(model, message_history)

            print(f"\nFinal token count: {final_tokens}")
            print(f"Maximum token limit: {max_tokens}")

        except Exception as error:
            print(f"\nError: {error}")

            # 9. Restore the previous conversation state.
            memory.replace_messages(
                session_id,
                previous_history
            )


def main():
    create_convo()


if __name__ == "__main__":
    main()
