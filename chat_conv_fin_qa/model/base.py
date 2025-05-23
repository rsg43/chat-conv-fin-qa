"""
Base module for the model wrapper, which provides a common interface for
interacting with different models. This module is designed to be extended
for specific models, such as OpenAI or Anthropic. The base class provides
methods for sending messages to the model, invoking the model with a
prompt, and binding tools to the model. The class also provides a method
for setting the output schema for the model.
"""

from typing import Any, Optional

from pydantic import BaseModel as PydanticBaseModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel


class BaseModel:
    """
    Base class for model wrappers. This class provides a common interface
    for interacting with different models.
    """

    _model: BaseChatModel
    _tools_bound: bool = False
    _tool_model: Optional[Runnable[LanguageModelInput, BaseMessage]]

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Method to send a list of messages to the model and generated the next
        response based on the messages.

        :param messages: List of messages to send to the model.
        :type messages: list[BaseMessage]
        :raises ValueError: If the model response is not an AIMessage.
        :return: The model's response message.
        :rtype: AIMessage
        """
        if self._tools_bound and self._tool_model:
            result = self._tool_model.invoke(input=messages)
        else:
            result = self._model.invoke(input=messages)

        if isinstance(result, AIMessage):
            return result
        else:
            raise ValueError("Model response is not an AIMessage.")

    def invoke(self, prompt: str) -> str:
        """
        Method to send a prompt to the model and get the response.

        :param prompt: The prompt to send to the model.
        :type prompt: str
        :raises ValueError: If the model response is not a string.
        :return: The model's response.
        :rtype: str
        """
        if self._tools_bound and self._tool_model:
            result = self._tool_model.invoke(input=prompt).content
        else:
            result = self._model.invoke(input=prompt).content

        if isinstance(result, str):
            return result
        else:
            raise ValueError("Model response is not a string.")

    def bind_tools(self, tools: list[dict[str, Any]]) -> None:
        """
        Method to bind tools to the model.

        :param tools: List of tool names to bind to the model.
        :type tools: list[dict[str, Any]]
        """
        self._tool_model = self._model.bind_tools(tools)
        self._tools_bound = True

    def with_structured_output(
        self, output_schema: dict[str, Any]
    ) -> Runnable[LanguageModelInput, dict[str, Any] | PydanticBaseModel]:
        """
        Method to set the output schema for the model.

        :param output_schema: The output schema to set.
        :type output_schema: dict[str, Any]
        :return: A runnable model with the specified output schema.
        :rtype: Runnable[LanguageModelInput, dict | PydanticBaseModel]
        """
        return self._model.with_structured_output(output_schema)
