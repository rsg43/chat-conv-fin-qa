"""
Module for Anthropic model integration. uses Clause 3 via LangChain Anthropic
chat model wrapper.
"""

from langchain_anthropic.chat_models import ChatAnthropic

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):
    """
    Anthropic model wrapper for LangChain Anthropic chat model.
    """

    _model = ChatAnthropic(
        model="claude-3-haiku-20240307",  # type: ignore[call-arg]
        temperature=0.3,
        max_tokens=1024,
        timeout=60,
        max_retries=3,
    )
