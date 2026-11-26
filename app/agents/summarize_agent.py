
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.agents.chat_agent import create_chat_model
from app.core.config import settings


SYS_PROMPT_SUMMARIZE = """
You are a conversation summarization assistant.

Produce a concise, structured summary that prioritizes durable context over ephemeral chatter. Organize relevant information into these categories (omit any category with no information):

1. Stable user facts: Name, occupation/role, location, background, and explicitly stated personal details.
2. User preferences: Stated communication preferences, tool/workflow choices, likes, and dislikes.
3. Technical skills: Programming languages, frameworks, tools, domains, and areas of expertise mentioned by the user.
4. Active projects: Current projects, goals, architectures, and ongoing work.
5. Decisions and commitments: Agreed decisions, action items, constraints, and next steps.
6. Temporary conversation details: Transient greetings, one-off clarifications, or short-lived context (condense heavily or omit unless needed for immediate continuity).

Strict grounding rules:
- Only record facts explicitly stated in the conversation.
- Never turn assumptions, guesses, or inferences into stated facts; if something is tentative or unconfirmed, mark it explicitly as tentative or omit it.
- Do not invent or extrapolate missing details.
"""

chat_summarize = create_chat_model(
    model_name=settings.chat_model
)


def summarize_agent(
    message_history: list[BaseMessage],
) -> str:
    """
    Summarize previous conversation messages.
    """

    transcript = "\n".join(
        f"{msg.type.upper()}: {msg.text}"
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

    return result.text
