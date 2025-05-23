# Chat ConvFinQA

# Requirements
Need to provide an Anthropic API key (or OpenAI, but I have not tested this!).
A dev container is provided which will installed requirements.

# Usage
I have included a very basic setup for a MCP client, which has use of a simple
maths MCP server (same tools as mentioned in the paper linked for the task).

To run the chat, use the command:
```bash
python3 -m chat_conv_fin_qa
```
This will allow you to provide a set of context data (or use a default) and
then ask it questions using the tools provided by the server.

There is also an interactive evaluation script which can be run using:
```bash
python3 -m chat_conv_fin_qa.mcp.client.evaluate
```
Currently, this provides the text and tables from the dataset as context, and
then uses the MCP client to answer, before using a further LLM call to rate the
quality of the answer against the reference answer.

# Future steps
I ran low on time due to work commitments, so would have like to extend the
evaluation to other metrics, and run on a larger sample of the data and produce
graphs (I don't have a GPU on my laptop, so this would've cost a fair bit in
tokens!). I would also have like to do more testing, and improve the prompting
strategy.

Ideally, I would have expanded on the prompts to do more chain of thoght
reasoning, and also tried one/few shot prompting to see if performance could be
improve for this particularly dataset. I would also have like to implement a
financial analysis MCP server to provide more in depth analysis. I could
potentially have expanded to use Flask/LangGraph as architecture became more
adavanced and when adding a frontend, but elected to prioritse time elsewhere.

Finally, I would have fixed all the linting issues, and set up more CI/CD in
GitHub, potentially creating docker images and packages on the container
repository.
