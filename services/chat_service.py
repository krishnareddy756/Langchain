from typing import Any, List, Optional, Sequence, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from models.chat_model import ModelFactory
from config.settings import Provider


class ChatService:
    """
    Provider-independent service responsible for interacting with chat models.
    Accepts any BaseChatModel or initializes one through the ModelFactory.
    """

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        provider: Optional[Union[Provider, str]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the service. If `model` is provided, it is used directly.
        Otherwise, ModelFactory is used to initialize the model based on provider/settings.
        """
        if model is not None:
            self._model: BaseChatModel = model
        else:
            self._model = ModelFactory.get_chat_model(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                **kwargs,
            )

    @property
    def model(self) -> BaseChatModel:
        return self._model

    def invoke_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs: Any,
    ) -> AIMessage:
        """Invokes the chat model with langchain-core BaseMessage abstractions."""
        return self._model.invoke(messages, **kwargs)

    def ask(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for text in / text out using System & Human messages."""
        messages: List[BaseMessage] = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        response = self.invoke_messages(messages, **kwargs)
        return str(response.content)

    def stream_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs: Any,
    ):
        """Streams model responses chunk-by-chunk."""
        return self._model.stream(messages, **kwargs)
