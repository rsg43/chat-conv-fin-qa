from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import BaseMessage, AIMessage


class BaseModel(ABC):

    @abstractmethod
    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Method to send a list of messages to the model and generated the next
        response based on the messages.

        :param messages: List of messages to send to the model.
        :type messages: list[BaseMessage]
        :return: The model's response message.
        :rtype: AIMessage
        """

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Method to send a prompt to the model and get the response.

        :param prompt: The prompt to send to the model.
        :type prompt: str
        :return: The model's response.
        :rtype: str
        """

    @abstractmethod
    def bind_tools(self, tools: list[dict[str, Any]]) -> None:
        """
        Method to bind tools to the model.

        :param tools: List of tool names to bind to the model.
        :type tools: list[dict[str, Any]]
        """
