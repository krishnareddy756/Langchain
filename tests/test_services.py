import unittest
from unittest.mock import MagicMock
from config.settings import Provider, Settings
from models.chat_model import ModelFactory
from services.chat_service import ChatService
from services.embedding_service import (
    EmbeddingService,
    EmbeddingError,
    DimensionMismatchError,
)
from langchain_core.messages import AIMessage, HumanMessage


class TestMultiProviderModelFactory(unittest.TestCase):

    def test_invalid_provider_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            ModelFactory.get_chat_model(provider="unsupported_provider")
        self.assertIn("Unsupported provider", str(ctx.exception))

    def test_provider_enum_from_str(self):
        self.assertEqual(Provider.from_str("google"), Provider.GOOGLE)
        self.assertEqual(Provider.from_str("OPENAI"), Provider.OPENAI)
        self.assertEqual(Provider.from_str("groq"), Provider.GROQ)

    def test_chat_service_is_provider_independent(self):
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="Hello from mock!")

        service = ChatService(model=mock_model)
        reply = service.ask("Hi")

        self.assertEqual(reply, "Hello from mock!")
        mock_model.invoke.assert_called_once()


class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        self.mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

    def test_query_and_document_embedding(self):
        service = EmbeddingService(
            embeddings=self.mock_embeddings,
            expected_dimension=3,
            config_version="v1.0.0",
        )
        self.assertEqual(service.config_version, "v1.0.0")

        q_vec = service.embed_query("test query")
        self.assertEqual(len(q_vec), 3)

        docs_vecs = service.embed_documents(["doc 1", "doc 2"])
        self.assertEqual(len(docs_vecs), 2)
        self.assertEqual(len(docs_vecs[0]), 3)

    def test_dimension_validation_failure(self):
        service = EmbeddingService(
            embeddings=self.mock_embeddings,
            expected_dimension=512,  # mismatch with 3
        )
        with self.assertRaises(DimensionMismatchError):
            service.embed_query("test query")

    def test_empty_query_validation(self):
        service = EmbeddingService(embeddings=self.mock_embeddings)
        with self.assertRaises(ValueError):
            service.embed_query("   ")

    def test_provider_failure_handling(self):
        failing_embeddings = MagicMock()
        failing_embeddings.embed_query.side_effect = RuntimeError("API Rate Limit")

        service = EmbeddingService(embeddings=failing_embeddings)
        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_query("test")
        self.assertIn("Embedding query failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

