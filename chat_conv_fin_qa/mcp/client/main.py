from typing import Any
from contextlib import AsyncExitStack
import asyncio
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolMessage,
)

from chat_conv_fin_qa.chat_history import ChatHistory
from chat_conv_fin_qa.model.anthropic import AnthropicModel


SYSTEM_PROMPT = (
    "You are a helpful assistant. Try to answer questions concisely and "
    "accuretely. You are also able to call tools to help you answer questions"
    ". Only use tools if the question requires it. "
)

SERVERS = {
    "maths": StdioServerParameters(
        command="uv",
        args=["run", "chat_conv_fin_qa/mcp/servers/maths.py"],
    )
}


class MCPClient:

    def __init__(self):
        self.chat_history = ChatHistory()
        self.sessions: list[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.tool_sessions: dict[str, ClientSession] = {}
        self._model = AnthropicModel()

    async def __aenter__(self):
        await self.connect_to_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect_to_servers(self):
        all_tools: list[dict[str, Any]] = []
        for _, server_params in SERVERS.items():
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(*stdio_transport)
            )
            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools

            for tool in tools:
                self.tool_sessions[tool.name] = session
                all_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                )

        self._model.bind_tools(all_tools)

    async def invoke(self, query: str, session_id: str):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + self.chat_history.get_messages(session_id)
        new_messages: list[BaseMessage] = [HumanMessage(content=query)]
        print("BEFORE")
        response = self._model.chat(messages + new_messages)
        print("AFTER")
        new_messages.append(response)

        while len(response.tool_calls) > 0:
            print(f"AI: {response.content[0]["text"]}")
            for tool_call in response.tool_calls:
                print(f"Tool call: {tool_call}")
                result = await self.tool_sessions[tool_call["name"]].call_tool(
                    name=tool_call["name"], arguments=tool_call["args"]
                )
                print(f"Result: {result.content}")

                new_messages.append(
                    ToolMessage(
                        content=result.content,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            response = self._model.chat(messages + new_messages)
            new_messages.append(response)

        print(f"AI: {response.content}")

        self.chat_history.add_messages(
            messages=new_messages, session_id=session_id
        )

    async def run(self):
        print("\nMCP Client Running! (enter q to quit)")
        session_id = uuid4().hex
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "q":
                    break

                await self.invoke(
                    query=query,
                    session_id=session_id,
                )
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    async with MCPClient() as client:
        await client.run()


if __name__ == "__main__":
    asyncio.run(main())
