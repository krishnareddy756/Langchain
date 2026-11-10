from typing import Any, Optional, Union
from langchain_core.embeddings import Embeddings
from config.settings import Provider, settings


class EmbeddingModelFactory:
    """
    Centralized factory for initializing embedding models across providers.
    """

    @staticmethod
    def get_embedding_model(
        provider: Optional[Union[Provider, str]] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Embeddings:
        resolved_provider: Provider
        if provider is None:
            resolved_provider = settings.DEFAULT_EMBEDDING_PROVIDER
        elif isinstance(provider, Provider):
            resolved_provider = provider
        elif isinstance(provider, str):
            resolved_provider = Provider.from_str(provider)
        else:
            raise ValueError(
                f"Invalid provider type: {type(provider)}. Must be str or Provider enum."
            )

        if resolved_provider == Provider.GOOGLE:
            return EmbeddingModelFactory._init_google_embeddings(
                model_name=model_name or settings.GOOGLE_EMBEDDING_MODEL,
                **kwargs,
            )
        elif resolved_provider == Provider.OPENAI:
            return EmbeddingModelFactory._init_openai_embeddings(
                model_name=model_name or settings.OPENAI_EMBEDDING_MODEL,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Provider '{resolved_provider.value}' does not support embeddings or is not implemented."
            )

    @staticmethod
    def _init_google_embeddings(
        model_name: str,
        **kwargs: Any,
    ) -> Embeddings:
        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "Missing GOOGLE_API_KEY. Set it in .env or your environment variables."
            )
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required. Run `uv add langchain-google-genai`."
            ) from e

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            **kwargs,
        )

    @staticmethod
    def _init_openai_embeddings(
        model_name: str,
        **kwargs: Any,
    ) -> Embeddings:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "Missing OPENAI_API_KEY. Set it in .env or your environment variables."
            )
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required. Run `uv add langchain-openai`."
            ) from e

        return OpenAIEmbeddings(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )

