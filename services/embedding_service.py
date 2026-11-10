import logging
from typing import Any, List, Optional, Union
from langchain_core.embeddings import Embeddings
from config.settings import Provider, settings
from embeddings.factory import EmbeddingModelFactory

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class DimensionMismatchError(EmbeddingError):
    """Raised when the vector dimension does not match expected dimension."""
    pass


class EmbeddingService:
    """
    Service for generating document and query embeddings with dimension validation,
    failure handling, and version tracking.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        provider: Optional[Union[Provider, str]] = None,
        model_name: Optional[str] = None,
        expected_dimension: Optional[int] = None,
        config_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param embeddings: An existing Embeddings instance.
        :param provider: "google" or "openai" (defaults to DEFAULT_EMBEDDING_PROVIDER).
        :param model_name: Name of the embedding model.
        :param expected_dimension: Optional dimension to strictly enforce/validate.
        :param config_version: Configuration version identifier.
        """
        self.config_version = config_version or settings.EMBEDDING_CONFIG_VERSION
        self.expected_dimension = expected_dimension
        self.provider = provider or settings.DEFAULT_EMBEDDING_PROVIDER

        if embeddings is not None:
            self._embeddings = embeddings
        else:
            self._embeddings = EmbeddingModelFactory.get_embedding_model(
                provider=provider,
                model_name=model_name,
                **kwargs,
            )

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    def validate_dimension(self, vector: List[float]) -> None:
        """Validates that the vector has the expected dimension if configured."""
        if not vector:
            raise EmbeddingError("Generated vector is empty.")
        if self.expected_dimension is not None and len(vector) != self.expected_dimension:
            raise DimensionMismatchError(
                f"Vector dimension mismatch: expected {self.expected_dimension}, "
                f"got {len(vector)} (config version: {self.config_version})."
            )

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        try:
            vector = self._embeddings.embed_query(text)
            self.validate_dimension(vector)
            return vector
        except DimensionMismatchError:
            raise
        except Exception as e:
            logger.error(f"Failed to embed query: {e}", exc_info=True)
            raise EmbeddingError(f"Embedding query failed: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.
        """
        if not texts:
            return []

        try:
            vectors = self._embeddings.embed_documents(texts)
            for vec in vectors:
                self.validate_dimension(vec)
            return vectors
        except DimensionMismatchError:
            raise
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}", exc_info=True)
            raise EmbeddingError(f"Embedding documents failed: {str(e)}") from e

