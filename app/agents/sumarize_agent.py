
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.agents.chat_agent import create_chat_model
from app.core.config import settings


SYS_PROMPT_SUMMARIZE = """
You are a conversation summarization assistant.

Summarize the conversation while preserving:
- User-provided facts and personal details.
- User preferences.
- Important decisions.
- Ongoing tasks and projects.
- Important questions and unresolved topics.

Do not invent information.
Distinguish facts from assumptions.
Keep the summary concise but informative.
"""

chat_summarize = create_chat_model(
    model_name=settings.chat_model
)


def sumarize_agent(
    message_history: list[BaseMessage],
) -> str:
    """
    Summarize previous conversation messages.
    """

    transcript = "\n".join(
        f"{msg.type.upper()}: {msg.content}"
        for msg in message_history
    )

    result = chat_summarize.invoke(
        [
            SystemMessage(content=SYS_PROMPT_SUMMARIZE),
            HumanMessage(
                content=(
                    "Summarize this conversation:\n\n"
                    f"{transcript}"
                )
            ),
        ]
    )

    return result.content
