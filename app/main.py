
from app.agents.chat_agent import create_chat_model

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    AIMessage,
)
from app.utils.chat_services import summarize_history
from app.core.config import settings

def count_token(model, messages):
    """
    Counts the number of tokens in the messages.
    """
    return model.get_num_tokens_from_messages(messages)

def create_convo():
    model = create_chat_model(settings.chat_model)

    message_history: list[BaseMessage] = [
        SystemMessage(
            content="You are a helpful assistant."
        )
    ]

    while True:
        text = input("\nYou: ").strip()

        if text.lower() in {"exit", "quit", "bye"}:
            print("Assistant: Goodbye!")
            break

        if not text:
            continue

        previous_length = len(message_history)
        message_history.append(
            HumanMessage(content=text)
        )

        print("\nAssistant: ", end="", flush=True)

        ai_response = ""

        try:
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

            message_history.append(
                AIMessage(content=ai_response)
            )
            tokens = count_token(model, message_history)
            print(f"Tokens: {tokens}")
            max_tokens = settings.sliding_window_size
            if tokens >= max_tokens:
                message_history = summarize_history(message_history, max_tokens, current_tokens=tokens)
            print("\n--- COMPRESSED HISTORY ---")

            for index, message in enumerate(message_history):
                print(
                    f"{index}: "
                    f"{message.type} -> {message.content}"
            )
            print(
                f"\nFinal token count: "
                f"{model.get_num_tokens_from_messages(message_history)}"
            )
            print(f"Maximum token limit: {settings.sliding_window_size}")

        except Exception as error:
            print(f"\nError: {error}")
            del message_history[previous_length:]


def main():
    create_convo()


if __name__ == "__main__":
    main()
