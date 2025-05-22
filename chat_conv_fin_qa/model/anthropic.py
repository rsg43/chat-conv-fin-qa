from langchain_anthropic.chat_models import ChatAnthropic

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):

    _model = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.3,
        max_tokens=1024,
        timeout=60,
        max_retries=3,
    )
