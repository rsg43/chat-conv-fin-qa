from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from chat_conv_fin_qa.model.anthropic import AnthropicModel
from chat_conv_fin_qa.chat_history import ChatHistory


def main() -> None:
    model = AnthropicModel()
    chat_history = ChatHistory()
    session_id = uuid4().hex
    prompt = (
        "You are a helpful assistant. Answer truthfully based on the "
        "information provided by the user, and try to keep responses to 3 "
        "sentences or less. "
    )

    while True:
        messages = [SystemMessage(content=prompt)] + chat_history.get_messages(
            session_id=session_id
        )
        human_message = HumanMessage(content=input("Human: ") or "Hello")
        messages.append(human_message)
        ai_message = model.chat(messages=messages)
        print(f"AI: {ai_message.content}")
        chat_history.add_messages(
            session_id=session_id, messages=[human_message, ai_message]
        )


if __name__ == "__main__":
    main()
