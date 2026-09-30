"""
Scripted stand-in for ``claude_agent_sdk.ClaudeSDKClient``.

Offline flow tests replace the real client with ``FakeClaudeSDKClient`` so the
public API can be exercised deterministically, without the Claude Code CLI or a
subscription. Real SDK message types are used, so the adapter's parsing logic is
tested against the actual data structures.
"""

import asyncio
from typing import Any, Dict, List, Optional

from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock, ThinkingBlock

DEFAULT_USAGE = {
    "input_tokens": 10,
    "cache_read_input_tokens": 5,
    "cache_creation_input_tokens": 0,
    "output_tokens": 7,
    "output_tokens_details": {"thinking_tokens": 3},
}


def text_delta(text: str, parent: Optional[str] = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        parent_tool_use_id=parent,
    )


def thinking_delta(thinking: str) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        },
    )


def assistant(
    text: Optional[str] = None, thinking: Optional[str] = None, model: str = "claude-test-1"
) -> AssistantMessage:
    content: List[Any] = []
    if thinking is not None:
        content.append(ThinkingBlock(thinking=thinking, signature="sig"))
    if text is not None:
        content.append(TextBlock(text=text))
    return AssistantMessage(content=content, model=model)


def result(
    is_error: bool = False,
    text: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    subtype: str = "success",
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=120,
        duration_api_ms=100,
        is_error=is_error,
        num_turns=1,
        session_id="session-123",
        stop_reason="end_turn",
        total_cost_usd=0.001,
        usage=DEFAULT_USAGE if usage is None else usage,
        result=text,
    )


def streamed_reply(*pieces: str, thinking: Optional[str] = None) -> List[Any]:
    """A typical partial-message stream: deltas, the full AssistantMessage, then the result."""
    messages: List[Any] = []
    if thinking:
        messages += [thinking_delta(thinking), assistant(thinking=thinking)]
    messages += [text_delta(p) for p in pieces]
    messages += [assistant(text="".join(pieces)), result()]
    return messages


class FakeClaudeSDKClient:
    """
    Mimics the subset of ClaudeSDKClient used by the adapter.

    Class-level attributes configure the next client; ``instances`` records every
    client created so tests can inspect options, prompts and lifecycle calls.
    """

    script: List[Any] = []
    connect_error: Optional[BaseException] = None
    delay: float = 0.0
    instances: List["FakeClaudeSDKClient"] = []

    def __init__(self, options: Any = None):
        self.options = options
        self.prompts: List[str] = []
        self.connected = False
        self.interrupted = False
        self.disconnected = False
        self.delivered = 0
        FakeClaudeSDKClient.instances.append(self)

    @classmethod
    def reset(cls, script: Optional[List[Any]] = None) -> None:
        cls.script = list(script or [])
        cls.connect_error = None
        cls.delay = 0.0
        cls.instances = []

    async def connect(self, prompt: Any = None) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.prompts.append(prompt)

    async def receive_response(self):
        for item in self.script:
            if self.interrupted:
                return
            if self.delay:
                await asyncio.sleep(self.delay)
            else:
                await asyncio.sleep(0)
            if isinstance(item, BaseException):
                raise item
            self.delivered += 1
            yield item

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        self.disconnected = True
