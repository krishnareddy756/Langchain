from langchain_core.messages import HumanMessage, SystemMessage
from services.chat_service import ChatService


def main():
    print("Initializing ChatService with Gemini...")
    service = ChatService()

    # Using langchain-core message abstractions
    messages = [
        SystemMessage(content="You are a helpful and concise AI assistant."),
        HumanMessage(content="Explain what LangChain is in one sentence."),
    ]

    print("\nSending prompt to model...")
    response = service.invoke_messages(messages)

    print("\n--- Model Response ---")
    print(response.content)


if __name__ == "__main__":
    main()

