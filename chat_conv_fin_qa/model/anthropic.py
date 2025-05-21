from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_anthropic.chat_models import ChatAnthropic
from mcp.types import Tool

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):

    def __init__(self):
        self._model = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.3,
            max_tokens=1024,
            timeout=60,
            max_retries=3,
        )
        self._tool_model: Optional[
            Runnable[LanguageModelInput, BaseMessage]
        ] = None
        self._tools_bound = False

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        if self._tools_bound and self._tool_model:
            return self._tool_model.invoke(input=messages)
        return self._model.invoke(input=messages)

    def invoke(self, prompt: str) -> str:
        if self._tools_bound and self._tool_model:
            return self._tool_model.invoke(
                input=[HumanMessage(content=prompt)]
            ).content
        return self._model.invoke(input=[HumanMessage(content=prompt)]).content

    def bind_tools(self, tools: list[Tool]) -> None:
        self._tool_model = self._model.bind_tools(tools)
        self._tools_bound = True
