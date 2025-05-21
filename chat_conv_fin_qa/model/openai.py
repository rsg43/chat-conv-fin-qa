from langchain_openai.chat_models import ChatOpenAI

from chat_conv_fin_qa.model.base import BaseModel


class AnthropicModel(BaseModel):

    _model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1024,
        timeout=60,
        max_retries=3,
    )
