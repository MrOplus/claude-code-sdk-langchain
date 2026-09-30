"""
Claude Code LangChain Adapter

Use Claude through your Claude Code subscription as a LangChain chat model
for prototyping, without per-token API charges.
"""

from .chat_model import (
    DEFAULT_MODEL,
    ClaudeCodeChatModel,
    ClaudeCodeError,
    ClaudeCodeTimeoutError,
)
from .message_converter import MessageConverter

__all__ = [
    "ClaudeCodeChatModel",
    "ClaudeCodeError",
    "ClaudeCodeTimeoutError",
    "DEFAULT_MODEL",
    "MessageConverter",
]

__version__ = "0.2.0"
