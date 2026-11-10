"""
Interactive demonstration of the Multi-Provider Model Factory and Embedding Service.
"""
from config.settings import Provider, settings
from services.chat_service import ChatService
from services.embedding_service import EmbeddingService


def demo_chat():
    print("=" * 60)
    print("1. Multi-Provider Chat Service Demo")
    print("=" * 60)

    # Example: Google Gemini
    print("\n-> Initializing Google Chat Model (gemini-3.5-flash)...")
    google_service = ChatService(provider=Provider.GOOGLE)
    reply = google_service.ask("State your model family and role in one brief sentence.")
    print("Response:\n", reply)


def demo_embeddings():
    print("\n" + "=" * 60)
    print("2. Embedding Service Demo")
    print("=" * 60)

    print(f"\n-> Initializing Embedding Service (version: {settings.EMBEDDING_CONFIG_VERSION})...")
    embedding_service = EmbeddingService(
        provider=Provider.GOOGLE,
        expected_dimension=3072,  # gemini-embedding-001 dimension
    )

    sample_query = "What is Retrieval-Augmented Generation?"
    print(f"Embedding query: '{sample_query}'")
    query_vector = embedding_service.embed_query(sample_query)
    print(f"Generated query vector! Dimension: {len(query_vector)}")

    sample_docs = [
        "LangChain helps orchestrate LLM workflows and prompts.",
        "Embeddings map semantic concepts into dense continuous vector spaces."
    ]
    print(f"\nEmbedding {len(sample_docs)} sample documents...")
    doc_vectors = embedding_service.embed_documents(sample_docs)
    print(f"Generated {len(doc_vectors)} document vectors! Dimensions: {[len(v) for v in doc_vectors]}")


def main():
    try:
        demo_chat()
        demo_embeddings()
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")


if __name__ == "__main__":
    main()
