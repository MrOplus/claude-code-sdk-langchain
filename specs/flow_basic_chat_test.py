"""
Live flow test: basic chat with ClaudeCodeChatModel against the real Claude Code CLI.
Reference: flow_basic_chat.md
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from claude_code_langchain import ClaudeCodeChatModel

from .test_helpers import get_test_model_name

pytestmark = pytest.mark.live


def test_basic_chat_invocation():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke([HumanMessage(content="Hello, who are you? Answer in one sentence.")])

    assert isinstance(response, AIMessage)
    assert response.content.strip(), "Response content is empty"
    assert len(response.content.split()) >= 3, f"Response too short: {response.content!r}"


async def test_async_chat_invocation():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = await model.ainvoke([HumanMessage(content="Explain LangChain in one sentence.")])

    assert isinstance(response, AIMessage)
    assert response.content.strip()


def test_response_carries_usage_and_model_metadata():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke("What is 2+2? Reply with just the number.")

    assert "4" in response.content
    assert response.usage_metadata is not None
    assert response.usage_metadata["output_tokens"] > 0
    assert response.usage_metadata["input_tokens"] > 0
    assert response.response_metadata["model_name"].startswith("claude-")
    assert response.response_metadata["session_id"]


def test_default_request_is_isolated_from_account_tools():
    """No built-in tools, MCP servers or account connectors are attached by default.
    Connector tool definitions would add thousands of input tokens per request."""
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke("Reply with just the word OK.")

    assert response.usage_metadata["input_tokens"] < 1500, response.usage_metadata


def test_system_message_is_followed():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke(
        [
            SystemMessage(content="Always start your answer with the exact word 'PYTHON:'"),
            HumanMessage(content="What is a list? One sentence."),
        ]
    )

    assert response.content.strip().startswith("PYTHON:"), response.content


def test_conversation_history_is_used():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke(
        [
            HumanMessage(content="My name is Alice."),
            AIMessage(content="Nice to meet you, Alice!"),
            HumanMessage(content="What is my name? Reply with just the name."),
        ]
    )

    assert "alice" in response.content.lower()


def test_stop_sequence_truncates_output():
    model = ClaudeCodeChatModel(model=get_test_model_name())

    response = model.invoke(
        "Count from 1 to 10 separated by single spaces. Output only the numbers.", stop=["6"]
    )

    assert "6" not in response.content
    assert "5" in response.content
    assert response.response_metadata["stop_reason"] == "stop_sequence"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
