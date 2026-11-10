import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def get_chat_model(
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Centralized factory for initializing Google Gemini chat models.
    
    Reads credentials and configurations securely from environment variables:
    - GOOGLE_API_KEY (or GEMINI_API_KEY)
    - GOOGLE_CHAT_MODEL (optional default model override)
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found. Please set 'GOOGLE_API_KEY' or 'GEMINI_API_KEY' "
            "in your environment or .env file."
        )

    resolved_model = (
        model_name
        or os.getenv("GOOGLE_CHAT_MODEL")
        or "gemini-3.5-flash"
    )

    return ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=temperature,
        google_api_key=api_key,
        **kwargs,
    )

