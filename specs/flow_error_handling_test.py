"""
Flow test: error handling in ClaudeCodeChatModel.
Errors are simulated with a scripted SDK client; everything is exercised through the
public API. Reference: flow_error_handling.md
"""

from unittest.mock import patch

import pytest
from claude_agent_sdk import CLIConnectionError, CLIJSONDecodeError, CLINotFoundError, ProcessError
from langchain_core.messages import HumanMessage

from claude_code_langchain import ClaudeCodeChatModel, ClaudeCodeError

from .fake_sdk import result, text_delta


def test_flow_sdk_not_installed_error():
    """Users without the SDK get installation instructions at construction time."""
    with patch("claude_code_langchain.chat_model.CLAUDE_CODE_AVAILABLE", False):
        with pytest.raises(ImportError) as exc_info:
            ClaudeCodeChatModel()

    message = str(exc_info.value)
    assert "claude-agent-sdk" in message
    assert "pip install" in message
    assert "@anthropic-ai/claude-code" in message


def test_flow_cli_not_found_error(fake_claude):
    """A missing CLI surfaces as ClaudeCodeError with install instructions."""
    fake_claude.connect_error = CLINotFoundError("Claude Code not found")

    with pytest.raises(ClaudeCodeError) as exc_info:
        ClaudeCodeChatModel().invoke("Hello")

    message = str(exc_info.value)
    assert "CLI not found" in message
    assert "npm install -g @anthropic-ai/claude-code" in message


def test_flow_process_error_reports_exit_code_and_stderr(fake_claude):
    fake_claude.script = [
        ProcessError("Claude Code process failed", exit_code=1, stderr="Unable to authenticate")
    ]

    with pytest.raises(ClaudeCodeError) as exc_info:
        ClaudeCodeChatModel().invoke([HumanMessage(content="Hello")])

    message = str(exc_info.value)
    assert "process error" in message.lower()
    assert "exit code 1" in message
    assert "authenticate" in message.lower()
    # Backward compatible: ClaudeCodeError is a RuntimeError
    assert isinstance(exc_info.value, RuntimeError)


def test_flow_json_decode_error_shows_offending_line(fake_claude):
    fake_claude.script = [
        CLIJSONDecodeError('{"content": "test", invalid}', ValueError("Expecting value"))
    ]

    with pytest.raises(ClaudeCodeError) as exc_info:
        ClaudeCodeChatModel().invoke("Test message")

    message = str(exc_info.value)
    assert "Failed to parse" in message
    assert "invalid" in message


def test_flow_connection_error(fake_claude):
    fake_claude.connect_error = CLIConnectionError("pipe closed")

    with pytest.raises(ClaudeCodeError, match="Could not connect"):
        ClaudeCodeChatModel().invoke("Hi")


def test_flow_error_result_from_cli(fake_claude):
    """An error result (e.g. unknown model) is reported with the CLI's explanation."""
    fake_claude.script = [
        result(is_error=True, text="There's an issue with the selected model", subtype="success")
    ]

    with pytest.raises(ClaudeCodeError, match="issue with the selected model"):
        ClaudeCodeChatModel(model="not-a-model").invoke("Hi")


async def test_flow_async_error_handling(fake_claude):
    fake_claude.connect_error = CLINotFoundError("Claude Code not found in PATH")

    with pytest.raises(ClaudeCodeError) as exc_info:
        await ClaudeCodeChatModel().ainvoke([HumanMessage(content="Async test")])

    assert "PATH" in str(exc_info.value)


def test_flow_streaming_error_after_partial_output(fake_claude):
    """Chunks received before a failure are delivered, then the error is raised."""
    fake_claude.script = [
        text_delta("Starting response..."),
        ProcessError("Connection lost", exit_code=2),
    ]

    received = []
    with pytest.raises(ClaudeCodeError, match="Connection lost"):
        for chunk in ClaudeCodeChatModel().stream("Stream test"):
            received.append(chunk.content)

    assert "".join(received) == "Starting response..."
    assert fake_claude.instances[0].disconnected


async def test_flow_async_streaming_error(fake_claude):
    fake_claude.script = [text_delta("partial"), ProcessError("boom", exit_code=3)]

    with pytest.raises(ClaudeCodeError, match="exit code 3"):
        async for _ in ClaudeCodeChatModel().astream("Hi"):
            pass


def test_flow_model_recovers_after_error(fake_claude):
    """A failed request does not leave the model in a broken state."""
    fake_claude.script = [ProcessError("temporary failure", exit_code=1)]
    model = ClaudeCodeChatModel()
    with pytest.raises(ClaudeCodeError):
        model.invoke("Hi")

    fake_claude.script = [text_delta("recovered"), result()]
    assert model.invoke("Hi").content == "recovered"


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [HumanMessage(content="")],
        [HumanMessage(content="   ")],
    ],
)
def test_flow_empty_input_is_rejected_before_calling_cli(fake_claude, messages):
    with pytest.raises(ValueError, match="(?i)empty|no valid message"):
        ClaudeCodeChatModel().invoke(messages)

    assert fake_claude.instances == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
