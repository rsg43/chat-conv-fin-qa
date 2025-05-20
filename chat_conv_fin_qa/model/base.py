from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage


class BaseModel(ABC):

    @abstractmethod
    def chat(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Method to send a list of messages to the model and generated the next
        response based on the messages.

        :param messages: List of messages to send to the model.
        :type messages: list[BaseMessage]
        :return: The model's response message.
        :rtype: BaseMessage
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
