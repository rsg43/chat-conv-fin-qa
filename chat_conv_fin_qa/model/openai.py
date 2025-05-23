"""
Module for OpenAI model integration. uses Clause 3 via LangChain OpenAI
chat model wrapper.
"""

from langchain_openai.chat_models import ChatOpenAI

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):
    """
    OpenAI model wrapper for LangChain OpenAI chat model.
    """

    _model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1024,  # type: ignore[call-arg]
        timeout=60,
        max_retries=3,
    )
