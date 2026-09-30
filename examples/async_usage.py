"""
Async usage examples for ClaudeCodeChatModel.

Every request runs in its own Claude Code CLI process, so independent requests
really do run in parallel when you use asyncio - a big win for fan-out
workloads, tool execution and agents.

Run with:  python examples/async_usage.py
"""

import asyncio
import time

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel

from claude_code_langchain import ClaudeCodeChatModel, ClaudeCodeTimeoutError

MODEL = "haiku"  # fast and light on quota; use "sonnet" or "opus" for harder tasks


# --------------------------------------------------------------------------- tools
# Async tools: the sleep simulates network I/O, so concurrent execution is visible.


@tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    await asyncio.sleep(1)
    return {"paris": "18C, light rain", "tokyo": "26C, sunny"}.get(city.lower(), "unknown")


@tool
async def get_time(city: str) -> str:
    """Get the current local time for a city."""
    await asyncio.sleep(1)
    return {"paris": "09:15", "tokyo": "16:15"}.get(city.lower(), "unknown")


TOOLS = {"get_weather": get_weather, "get_time": get_time}


# ------------------------------------------------------------------------ examples


async def example_ainvoke():
    """Example 1: single async invocation"""
    print("\n📝 Example 1: ainvoke")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    response = await model.ainvoke("What is asyncio, in one sentence?")

    print(f"Response: {response.content}")
    print(f"Tokens:   {response.usage_metadata}")


async def example_astream():
    """Example 2: token-by-token async streaming"""
    print("\n🌊 Example 2: astream")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    print("Streaming: ", end="")
    async for chunk in model.astream("Write a haiku about concurrency."):
        print(chunk.content, end="", flush=True)
    print()


async def example_concurrent_requests():
    """Example 3: independent requests in parallel with asyncio.gather"""
    print("\n🚀 Example 3: Concurrent Requests")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    questions = [
        "Capital of France? One word.",
        "Capital of Japan? One word.",
        "Capital of Brazil? One word.",
    ]

    start = time.monotonic()
    responses = await asyncio.gather(*(model.ainvoke(q) for q in questions))
    elapsed = time.monotonic() - start

    for question, response in zip(questions, responses):
        print(f"{question} -> {response.content.strip()}")
    print(f"{len(questions)} requests in {elapsed:.1f}s (in parallel, not one after another)")

    # abatch does the same, with an optional concurrency limit
    answers = await model.abatch(questions, config={"max_concurrency": 2})
    print(f"abatch: {[a.content.strip() for a in answers]}")


async def example_async_chain():
    """Example 4: async LCEL chain, streamed through an output parser"""
    print("\n🔗 Example 4: Async Chain Streaming")
    print("-" * 40)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an expert in {domain}. Answer in two sentences."), ("human", "{q}")]
    )
    chain = prompt | ClaudeCodeChatModel(model=MODEL) | StrOutputParser()

    print("Streaming: ", end="")
    async for text in chain.astream({"domain": "networking", "q": "What is a socket?"}):
        print(text, end="", flush=True)  # the parser yields plain strings
    print()


async def example_async_tool_loop():
    """Example 5: async tool loop - requested tools run concurrently"""
    print("\n🛠️ Example 5: Async Tool Loop")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL).bind_tools([get_weather, get_time])
    messages: list[BaseMessage] = [
        HumanMessage(content="What's the weather and the local time in Tokyo?")
    ]

    while True:
        ai = await model.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break

        # Run all requested tools at once; each takes ~1s, together still ~1s
        start = time.monotonic()
        results = await asyncio.gather(
            *(TOOLS[call["name"]].ainvoke(call["args"]) for call in ai.tool_calls)
        )
        names = [f"{c['name']}({c['args']['city']})" for c in ai.tool_calls]
        print(f"  ran {names} in {time.monotonic() - start:.1f}s")

        messages += [
            ToolMessage(result, tool_call_id=call["id"])
            for call, result in zip(ai.tool_calls, results)
        ]

    print(f"Answer: {ai.content}")


async def example_async_structured_output():
    """Example 6: structured output, extracted concurrently"""
    print("\n🧾 Example 6: Async Structured Output")
    print("-" * 40)

    class Person(BaseModel):
        """Information about a person."""

        name: str
        age: int

    extractor = ClaudeCodeChatModel(model=MODEL).with_structured_output(Person)
    texts = ["Maria Lopez is 34.", "Kenji Sato turned 51 last week.", "Ada, aged 36, codes."]

    people = await asyncio.gather(*(extractor.ainvoke(t) for t in texts))
    for person in people:
        print(f"  {person!r}")


async def example_async_agent():
    """Example 7: LangChain agent (LangGraph), invoked and streamed asynchronously"""
    print("\n🤖 Example 7: Async Agent")
    print("-" * 40)

    try:
        from langchain.agents import create_agent
    except ImportError:
        print("  skipped: pip install langchain")
        return

    agent = create_agent(
        ClaudeCodeChatModel(model=MODEL),
        tools=[get_weather, get_time],
        system_prompt="You are a concise travel assistant.",
    )

    question = HumanMessage(content="Compare the weather in Paris and Tokyo.")
    async for update in agent.astream({"messages": [question]}, stream_mode="updates"):
        for node, state in update.items():
            for message in state.get("messages", []):
                calls = [c["name"] for c in getattr(message, "tool_calls", [])]
                label = f"calls {calls}" if calls else repr(str(message.content)[:70])
                print(f"  [{node}] {type(message).__name__}: {label}")


async def example_timeout_and_cancellation():
    """Example 8: timeouts and stopping a stream early"""
    print("\n⏱️ Example 8: Timeout and Cancellation")
    print("-" * 40)

    # A per-request timeout raises ClaudeCodeTimeoutError (also a TimeoutError)
    impatient = ClaudeCodeChatModel(model=MODEL, timeout=2)
    try:
        await impatient.ainvoke("Write a 1,000-word essay about the history of computing.")
    except ClaudeCodeTimeoutError as e:
        print(f"  timeout: {e}")

    # Breaking out of a stream stops generation - the CLI request is interrupted
    model = ClaudeCodeChatModel(model=MODEL)
    received = ""
    async for chunk in model.astream("Count slowly from 1 to 100, one number per line."):
        received += str(chunk.content)
        if len(received) > 20:
            break
    print(f"  stopped early after: {received.strip()!r}")

    # asyncio.wait_for works too
    try:
        await asyncio.wait_for(model.ainvoke("Write a long poem about the sea."), timeout=2)
    except TimeoutError:
        print("  asyncio.wait_for cancelled the request")


async def main():
    print("=" * 50)
    print("⚡ Async ClaudeCodeChatModel Examples")
    print("=" * 50)

    await example_ainvoke()
    await example_astream()
    await example_concurrent_requests()
    await example_async_chain()
    await example_async_tool_loop()
    await example_async_structured_output()
    await example_async_agent()
    await example_timeout_and_cancellation()

    print("\n" + "=" * 50)
    print("✅ All async examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
