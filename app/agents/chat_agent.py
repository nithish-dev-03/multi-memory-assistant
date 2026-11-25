from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

def create_chat_model(model_name: str):
    """
    Creates a chat model instance using the settings.
    """
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    return ChatGoogleGenerativeAI(
        model=model_name,        
    )
    