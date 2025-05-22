from typing import Any
from asyncio import run
import json
from uuid import uuid4
from textwrap import dedent

from chat_conv_fin_qa.mcp.client.main import MCPClient, SYSTEM_PROMPT_TEMPLATE

CONTEXT_TEMPLATE = dedent(
    """
    Pre text:
    {pre_text}

    Post text:
    {post_text}

    Table:
    {table}
    """
)


class EvaluateClient(MCPClient):

    def __init__(self):
        super().__init__()

    async def evaluate(self):
        with open("data/train.json", "r") as f:
            data: list[dict[str, Any]] = json.load(f)

        cnt = 0
        for item in data:
            if (
                "qa" not in item
                or "question" not in item["qa"]
                or "answer" not in item["qa"]
                or "explanation" not in item["qa"]
            ):
                continue

            self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                context=CONTEXT_TEMPLATE.format(
                    pre_text=item["pre_text"],
                    post_text=item["post_text"],
                    table=item["table"],
                )
            )
            print(f"Question: {item['qa']['question']}")
            response = await self.invoke(item["qa"]["question"], uuid4().hex)
            print("-" * 79)
            print(f"AI Answer: {response.content}")
            print(f"Expected Answer: {item['qa']['answer']}")
            print(f"Explanation: {item['qa']['explanation']}")
            print("=" * 79)

            cnt += 1
            if cnt > 10:
                break


async def main() -> None:
    async with EvaluateClient() as client:
        await client.evaluate()


if __name__ == "__main__":
    run(main())
