"""
Helpers for live flow tests.
"""

import os

DEFAULT_TEST_MODEL = "haiku"


def get_test_model_name() -> str:
    """
    Model used by live flow tests.

    Reads ``CLAUDE_TEST_MODEL`` and defaults to ``haiku`` (fastest, lowest quota use).

    Usage:
        model = ClaudeCodeChatModel(model=get_test_model_name())
    """
    return os.environ.get("CLAUDE_TEST_MODEL", DEFAULT_TEST_MODEL)
