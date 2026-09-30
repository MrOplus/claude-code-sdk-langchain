"""
Shared pytest configuration for flow tests.

Two kinds of flow tests live in this directory:

- Offline flows use the ``fake_claude`` fixture (a scripted SDK client). They are
  fast and deterministic and need neither the CLI nor a subscription.
- Live flows (marked ``live``) call the real Claude Code CLI. They are skipped
  automatically when the CLI is not installed, or when ``CLAUDE_SKIP_LIVE=1``.

Run only offline tests with:  pytest -m "not live"
"""

import os
import shutil

import pytest

from .fake_sdk import FakeClaudeSDKClient


def pytest_collection_modifyitems(config, items):
    reason = None
    if os.environ.get("CLAUDE_SKIP_LIVE") == "1":
        reason = "CLAUDE_SKIP_LIVE=1"
    elif shutil.which("claude") is None:
        reason = "Claude Code CLI not found on PATH"

    if reason:
        skip_live = pytest.mark.skip(reason=f"live test skipped: {reason}")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture
def fake_claude(monkeypatch):
    """Replace the SDK client with a scripted fake. Set ``fake_claude.script`` to the
    messages (or exceptions) the next request should produce."""
    FakeClaudeSDKClient.reset()
    monkeypatch.setattr("claude_code_langchain.chat_model.ClaudeSDKClient", FakeClaudeSDKClient)
    yield FakeClaudeSDKClient
    FakeClaudeSDKClient.reset()
