"""
Live flow test: streaming responses from ClaudeCodeChatModel.
Reference: flow_streaming.md
"""

import time

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from claude_code_langchain import ClaudeCodeChatModel

from .test_helpers import get_test_model_name

pytestmark = pytest.mark.live


@pytest.fixture
def model():
    return ClaudeCodeChatModel(model=get_test_model_name())


def test_flow_sync_streaming_is_incremental(model):
    chunks = list(model.stream([HumanMessage(content="Count from 1 to 30, separated by commas.")]))

    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    text_chunks = [c for c in chunks if c.content]
    assert len(text_chunks) >= 2, "Expected token-level streaming, got a single block"
    assert "30" in "".join(c.content for c in text_chunks)


async def test_flow_async_streaming_is_incremental(model):
    arrival_times = []
    content = ""
    start = time.monotonic()

    async for chunk in model.astream("Explain async programming in 3 short numbered steps."):
        if chunk.content:
            arrival_times.append(time.monotonic() - start)
            content += chunk.content

    assert content.strip()
    assert len(arrival_times) >= 2
    assert arrival_times[-1] > arrival_times[0], "Chunks should arrive over time"


def test_flow_streamed_chunks_aggregate_to_full_message(model):
    total = None
    for chunk in model.stream("What is 10 + 15? Reply with just the number."):
        total = chunk if total is None else total + chunk

    assert total is not None
    assert "25" in total.content
    assert total.usage_metadata is not None
    assert total.usage_metadata["output_tokens"] > 0


def test_flow_chain_streaming(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a storyteller. Tell a story about {topic} in 3 sentences."),
            ("human", "{request}"),
        ]
    )

    story = "".join(
        chunk.content
        for chunk in (prompt | model).stream(
            {"topic": "a brave robot", "request": "Make it inspiring"}
        )
    )

    assert "robot" in story.lower(), "Chain didn't pass the topic variable"


async def test_flow_async_chain_streaming_with_parser(model):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant"), ("human", "{question}")]
    )
    chain = prompt | model | StrOutputParser()

    parts = [part async for part in chain.astream({"question": "What is the capital of France?"})]

    assert all(isinstance(p, str) for p in parts)
    assert "paris" in "".join(parts).lower()


def test_flow_streaming_cancellation(model):
    collected = []
    for chunk in model.stream("Count from 1 to 100 slowly, explaining each number."):
        if chunk.content:
            collected.append(chunk.content)
        if len(collected) >= 3:
            break

    assert len(collected) == 3

    # The model is still usable after an abandoned stream
    assert model.invoke("Reply with just the word OK.").content.strip()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
