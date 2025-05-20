from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic.chat_models import ChatAnthropic

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

    def chat(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._model.invoke(input=messages)

    def invoke(self, prompt: str) -> str:
        return self._model.invoke(input=[HumanMessage(content=prompt)]).content
