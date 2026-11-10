from typing import Any, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import Provider, settings


class ModelFactory:
    """
    Centralized Multi-Provider Model Factory.
    Initializes BaseChatModel instances for Google, OpenAI, Groq, etc.
    """

    @staticmethod
    def get_chat_model(
        provider: Optional[Union[Provider, str]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseChatModel:
        """
        Creates and returns a chat model instance conforming to BaseChatModel.

        :param provider: "google", "openai", "groq" or Provider enum. Defaults to DEFAULT_CHAT_PROVIDER.
        :param model_name: Name of the model. Defaults to provider-specific configuration.
        :param temperature: Model sampling temperature.
        :param kwargs: Additional provider-specific keyword arguments.
        """
        resolved_provider: Provider
        if provider is None:
            resolved_provider = settings.DEFAULT_CHAT_PROVIDER
        elif isinstance(provider, Provider):
            resolved_provider = provider
        elif isinstance(provider, str):
            resolved_provider = Provider.from_str(provider)
        else:
            raise ValueError(
                f"Invalid provider type: {type(provider)}. Must be str or Provider enum."
            )

        if resolved_provider == Provider.GOOGLE:
            return ModelFactory._init_google_chat(
                model_name=model_name or settings.GOOGLE_CHAT_MODEL,
                temperature=temperature,
                **kwargs,
            )
        elif resolved_provider == Provider.OPENAI:
            return ModelFactory._init_openai_chat(
                model_name=model_name or settings.OPENAI_CHAT_MODEL,
                temperature=temperature,
                **kwargs,
            )
        elif resolved_provider == Provider.GROQ:
            return ModelFactory._init_groq_chat(
                model_name=model_name or settings.GROQ_CHAT_MODEL,
                temperature=temperature,
                **kwargs,
            )
        else:
            raise ValueError(f"Unhandled provider: {resolved_provider}")

    @staticmethod
    def _init_google_chat(
        model_name: str,
        temperature: float,
        **kwargs: Any,
    ) -> BaseChatModel:
        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "Missing GOOGLE_API_KEY. Set it in .env or your environment variables."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Google models. Run `uv add langchain-google-genai`."
            ) from e

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            **kwargs,
        )

    @staticmethod
    def _init_openai_chat(
        model_name: str,
        temperature: float,
        **kwargs: Any,
    ) -> BaseChatModel:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "Missing OPENAI_API_KEY. Set it in .env or your environment variables."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI models. Run `uv add langchain-openai`."
            ) from e

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )

    @staticmethod
    def _init_groq_chat(
        model_name: str,
        temperature: float,
        **kwargs: Any,
    ) -> BaseChatModel:
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "Missing GROQ_API_KEY. Set it in .env or your environment variables."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError as e:
            raise ImportError(
                "langchain-groq is required for Groq models. Run `uv add langchain-groq`."
            ) from e

        return ChatGroq(
            model_name=model_name,
            temperature=temperature,
            api_key=settings.GROQ_API_KEY,
            **kwargs,
        )


# Backward-compatible helper
def get_chat_model(
    provider: Optional[Union[Provider, str]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    return ModelFactory.get_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        **kwargs,
    )
