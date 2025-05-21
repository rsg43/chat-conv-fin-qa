from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):

    def __init__(self) -> None:
        self._model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1024,
            timeout=60,
            max_retries=3,
        )

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        return self._model.invoke(input=messages)

    def invoke(self, prompt: str) -> str:
        return self._model.invoke(input=[HumanMessage(content=prompt)]).content
