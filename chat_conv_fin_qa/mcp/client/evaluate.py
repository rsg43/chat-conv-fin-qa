"""
Module to evaluate the MCP client using a set of questions and answers from the
Conversational Financial Question Answering (CFQA) dataset. The evaluation
process involves checking the answers provided by the model against the
expected answers and scoring them based on their correctness.

The evaluation is done using a structured output model, which allows for
validation of the scores returned by the model and accounting for any verbose
answers which have numerical answers, negatives, differences etc.

This could be extended to use other evaluation metrics such as BLEU or ROUGE
scores, but for now we are using a simple scoring system from 0 to 100.
"""

from typing import Any
from asyncio import run
import json
from uuid import uuid4
from textwrap import dedent

from pydantic import BaseModel

from chat_conv_fin_qa.mcp.client.main import MCPClient, SYSTEM_PROMPT_TEMPLATE
from chat_conv_fin_qa.model.anthropic import AnthropicModel


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


class Score(BaseModel):
    score: int


EVALUATION_PROMPT = dedent(
    """
    You are an exam checker. you will be given a question and an answer, along
    with the correct reference answer. Your task is to evaluate the answer and
    give a score from 0 to 100. 0 means the answer is completely wrong, and 100
    means the answer is completely correct. If the answer is partially correct,
    give a score between 0 and 100.

    <answer>{answer}</answer>

    <reference_answer>{reference}</reference_answer>

    Please use the schema below to give your score:

    <schema>{schema}</schema>
    """
)


class EvaluateClient(MCPClient):
    """
    Client to evaluate the MCP client using a set of questions and answers.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_structured = AnthropicModel().with_structured_output(
            output_schema=Score.model_json_schema()
        )

    async def evaluate(self) -> None:
        """
        Evaluate the MCP client using a set of questions and answers from the
        Conversational Financial Question Answering (CFQA) dataset. The
        evaluation process involves checking the answers provided by the model
        against the expected answers and scoring them based on their
        correctness.
        """
        with open("data/train.json", "r") as f:
            data: list[dict[str, Any]] = json.load(f)

        cnt = 0
        for item in data:
            if (
                "qa" not in item
                or "question" not in item["qa"]
                or "answer" not in item["qa"]
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

            structred_response = self._model_structured.invoke(
                input=EVALUATION_PROMPT.format(
                    answer=response.content,
                    reference=item["qa"]["answer"],
                    schema=Score.model_json_schema(),
                )
            )
            try:
                score = Score.model_validate(structred_response)
                print(f"Score: {score.score}")
            except Exception:
                print("Score validation error")
            print("=" * 79)

            cnt += 1
            if cnt > 10:
                break


async def main() -> None:
    """
    Main function to run the evaluation client.
    """
    async with EvaluateClient() as client:
        await client.evaluate()


if __name__ == "__main__":
    run(main())
