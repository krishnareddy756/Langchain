from typing import Any, List, Optional, Sequence, Union
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from models.chat_model import get_chat_model


class ChatService:
    """
    Service responsible for invoking the chat model using langchain-core message abstractions.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        """
        Initialize the service with an optional pre-configured model,
        or default to the centralized model initialization.
        """
        self._model = model or get_chat_model()

    @property
    def model(self):
        return self._model

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs,
    ) -> AIMessage:
        """
        Invokes the model with a sequence of BaseMessage abstractions.
        """
        response = self._model.invoke(messages, **kwargs)
        return response

    def ask(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Convenience method to query the model with user text and optional system instruction.
        Converts inputs into langchain-core SystemMessage and HumanMessage.
        """
        messages: List[BaseMessage] = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        response: AIMessage = self.invoke_messages(messages, **kwargs)
        return str(response.content)

    def stream_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs,
    ):
        """
        Streams model responses chunk-by-chunk using langchain-core BaseMessage.
        """
        return self._model.stream(messages, **kwargs)

