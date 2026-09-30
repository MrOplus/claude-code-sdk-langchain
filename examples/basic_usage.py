"""
Usage examples for ClaudeCodeChatModel with LangChain.

Shows how to use Claude through your Claude Code subscription to prototype
LangChain applications WITHOUT per-token API charges.

Run with:  python examples/basic_usage.py
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from claude_code_langchain import ClaudeCodeChatModel

MODEL = "haiku"  # fast and light on quota; use "sonnet" or "opus" for harder tasks


def example_simple_invocation():
    """Example 1: simple invocation with usage metadata"""
    print("\n📝 Example 1: Simple Invocation")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    response = model.invoke("What is LangChain, in 2 sentences?")

    print(f"Response: {response.content}")
    print(f"Model:    {response.response_metadata['model_name']}")
    print(f"Tokens:   {response.usage_metadata}")


def example_with_system_prompt():
    """Example 2: system prompts"""
    print("\n🎯 Example 2: System Prompt")
    print("-" * 40)

    model = ClaudeCodeChatModel(
        model=MODEL,
        system_prompt="You are a Python expert who answers very concisely.",
    )

    # A SystemMessage in the input takes precedence over the constructor prompt
    messages = [
        SystemMessage(content="Include a short code example when relevant."),
        HumanMessage(content="How do I create a list in Python?"),
    ]
    print(f"Response: {model.invoke(messages).content}")


def example_streaming():
    """Example 3: token-by-token streaming"""
    print("\n🌊 Example 3: Streaming")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)

    print("Streaming: ", end="")
    for chunk in model.stream("Tell a very short story about a robot."):
        print(chunk.content, end="", flush=True)
    print()


def example_langchain_chain():
    """Example 4: LCEL chain"""
    print("\n🔗 Example 4: LangChain Chain")
    print("-" * 40)

    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are an expert assistant in {domain}."), ("human", "{question}")]
    )
    chain = prompt | ClaudeCodeChatModel(model=MODEL) | StrOutputParser()

    response = chain.invoke(
        {"domain": "artificial intelligence", "question": "What is a neural network?"}
    )
    print(f"Response: {response}")


async def example_async_operations():
    """Example 5: async invocation and streaming"""
    print("\n⚡ Example 5: Async Operations")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)

    response = await model.ainvoke("What is Python async/await? One paragraph.")
    print(f"Async response: {response.content[:200]}...")

    print("\nAsync streaming: ", end="")
    async for chunk in model.astream("List 3 advantages of async programming."):
        print(chunk.content, end="", flush=True)
    print()


def example_batch_processing():
    """Example 6: batch processing"""
    print("\n📦 Example 6: Batch Processing")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    questions = [
        "Capital of France? One word.",
        "Capital of Spain? One word.",
        "Capital of Italy? One word.",
    ]

    for question, response in zip(questions, model.batch(questions)):
        print(f"{question} -> {response.content}")


def example_routing_chain():
    """Example 7: two-step chain that routes on a classification"""
    print("\n🤖 Example 7: Routing Chain")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)

    classify = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Answer with exactly one word: technical or general."),
                ("human", "{question}"),
            ]
        )
        | model
        | StrOutputParser()
    )

    answer = ChatPromptTemplate.from_messages(
        [("system", "Answer this question {style}."), ("human", "{question}")]
    )

    def route(inputs: dict) -> dict:
        kind = classify.invoke({"question": inputs["question"]})
        style = "in technical depth" if "technical" in kind.lower() else "in simple terms"
        return {"style": style, "question": inputs["question"]}

    chain = RunnableLambda(route) | answer | model | StrOutputParser()
    response = chain.invoke({"question": "How does a REST API work?"})
    print(f"Adapted response: {response[:300]}...")


def example_conversation_history():
    """Example 8: multi-turn conversation (stateless model, explicit history)"""
    print("\n💭 Example 8: Multi-Turn Conversation")
    print("-" * 40)

    # ClaudeCodeChatModel is stateless, like every LangChain chat model:
    # conversation context is carried by the messages you pass in.
    model = ClaudeCodeChatModel(model=MODEL)
    conversation = [HumanMessage(content="My name is Alice.")]

    reply = model.invoke(conversation)
    print(f"R1: {reply.content}")

    conversation += [reply, HumanMessage(content="What is my name?")]
    print(f"R2: {model.invoke(conversation).content}")


def example_stop_sequences():
    """Example 9: stop sequences (emulated client-side)"""
    print("\n✋ Example 9: Stop Sequences")
    print("-" * 40)

    model = ClaudeCodeChatModel(model=MODEL)
    response = model.invoke("Count from 1 to 10 separated by spaces.", stop=["6"])
    print(
        f"Truncated: {response.content!r} (stop_reason={response.response_metadata['stop_reason']})"
    )


def main():
    print("=" * 50)
    print("🚀 ClaudeCodeChatModel Examples for LangChain")
    print("=" * 50)
    print("\nUses your Claude Code subscription - no per-token API charges.")

    example_simple_invocation()
    example_with_system_prompt()
    example_streaming()
    example_langchain_chain()
    example_batch_processing()
    example_routing_chain()
    asyncio.run(example_async_operations())
    example_conversation_history()
    example_stop_sequences()

    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("=" * 50)
    print("\n💡 Tip: swap ClaudeCodeChatModel for ChatAnthropic when you are")
    print("   ready for production with the official API.")


if __name__ == "__main__":
    main()
