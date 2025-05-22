from typing import Any
from contextlib import AsyncExitStack
from asyncio import run
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
    ToolMessage,
)
from mcp.types import TextContent

from chat_conv_fin_qa.chat_history import ChatHistory
from chat_conv_fin_qa.model.anthropic import AnthropicModel

DEFAULT_CONTEXT = """
 [
    [
        "2008",
        "year ended june 30 2009 2008",
        "year ended june 30 2009 2008",
        "year ended june 30 2009"
    ],
    [
        "net income",
        "$ 103102",
        "$ 104222",
        "$ 104681"
    ],
    [
        "non-cash expenses",
        "74397",
        "70420",
        "56348"
    ],
    [
        "change in receivables",
        "21214",
        "-2913 ( 2913 )",
        "-28853 ( 28853 )"
    ],
    [
        "change in deferred revenue",
        "21943",
        "5100",
        "24576"
    ],
    [
        "change in other assets and liabilities",
        "-14068 ( 14068 )",
        "4172",
        "17495"
    ],
    [
        "net cash from operating activities",
        "$ 206588",
        "$ 181001",
        "$ 174247"
    ]
]
"""


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Try to answer questions concisely and "
    "accuretely. You are also able to call tools to help you answer questions"
    ". Only use tools if the question requires it. Answer the questions based"
    "on the conversation history and the context below:\n\n"
    "<context>{context}</context>\n\n"
)

SERVERS = {
    "maths": StdioServerParameters(
        command="uv",
        args=["run", "chat_conv_fin_qa/mcp/servers/maths.py"],
    )
}


class MCPClient:

    def __init__(self) -> None:
        self.chat_history = ChatHistory()
        self.sessions: list[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.tool_sessions: dict[str, ClientSession] = {}
        self._model = AnthropicModel()
        self._system_prompt: str

    async def __aenter__(self):
        await self.connect_to_servers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect_to_servers(self) -> None:
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

    async def invoke(self, query: str, session_id: str) -> None:
        messages = [
            SystemMessage(content=self._system_prompt)
        ] + self.chat_history.get_messages(session_id)
        new_messages: list[BaseMessage] = [HumanMessage(content=query)]
        response = self._model.chat(messages + new_messages)
        new_messages.append(response)

        while len(response.tool_calls) > 0:
            print(f"AI: {response.content[0]["text"]}")
            for tool_call in response.tool_calls:
                print(
                    f'Tool: {tool_call["name"]}, arguments {tool_call["args"]}'
                )
                result = await self.tool_sessions[tool_call["name"]].call_tool(
                    name=tool_call["name"], arguments=tool_call["args"]
                )
                print_result = (
                    val.text
                    if isinstance(val := result.content[0], TextContent)
                    else val
                )
                print(f"Result: {print_result}")

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

    async def run(self) -> None:
        print("\nMCP Client Running! (enter q to quit)")

        print("Please enter the context you want to query against. ")
        context = input("Context: ").strip()
        if not context:
            print("using default context")
            context = DEFAULT_CONTEXT
            print(context)

        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
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

    async def close(self) -> None:
        await self.exit_stack.aclose()


async def main() -> None:
    async with MCPClient() as client:
        await client.run()


def start_client() -> None:
    run(main())
