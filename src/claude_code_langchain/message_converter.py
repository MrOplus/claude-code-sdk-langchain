"""
Message conversion between LangChain and the Claude Agent SDK.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata

logger = logging.getLogger(__name__)

_IMAGE_PART_TYPES = {"image_url", "image", "input_image"}

_AFTER_TOOL_RESULTS = (
    "The tool results above are the outputs of the tool calls you (the assistant) already "
    "made. Do not repeat those calls. Continue your reply to the user using the results, "
    "calling a tool only for information that has not been provided yet. Answer naturally, "
    "without referring to this conversation format or these instructions."
)


class MessageConverter:
    """Converts between LangChain messages and Claude Agent SDK inputs/outputs."""

    @staticmethod
    def content_to_text(content: Any, index: int = 0) -> str:
        """
        Extract plain text from a LangChain message content value.

        Strings are returned as-is (stripped). For multimodal content lists, text parts
        are joined and non-text parts (images, files, ...) are dropped with a warning,
        because the Claude Code CLI prompt channel is text-only.

        Args:
            content: Message content (str, list of parts, or other)
            index: Message position, used in warning messages

        Returns:
            The extracted text (possibly empty)
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()

        text_parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue

            part_type = part.get("type", "")
            if part_type in _IMAGE_PART_TYPES or "image_url" in part:
                logger.warning(
                    f"Image content detected in message {index} but NOT SUPPORTED by the "
                    "Claude Code adapter. The image will be ignored. This differs from "
                    "production API behavior (ChatAnthropic supports vision)."
                )
            elif "text" in part and part_type in ("", "text"):
                text_parts.append(str(part["text"]))
            elif part_type:
                logger.warning(
                    f"Non-text content type '{part_type}' detected in message {index} and "
                    "will be ignored. Only text content is supported."
                )

        return "\n".join(p for p in text_parts if p).strip()

    @classmethod
    def message_text(cls, message: BaseMessage, index: int = 0) -> str:
        """
        Text of a message as it appears in the prompt.

        AI messages include their tool calls, so a conversation that used tools can be
        replayed to the CLI (which starts a fresh session for every request).
        """
        text = cls.content_to_text(message.content, index)
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = [
                f"[Tool call {call.get('id') or ''}] {call['name']}({json.dumps(call['args'])})"
                for call in message.tool_calls
            ]
            text = "\n".join(([text] if text else []) + calls)
        return text

    @classmethod
    def split_messages(cls, messages: List[BaseMessage]) -> Tuple[Optional[str], str]:
        """
        Split LangChain messages into a system prompt and a user prompt.

        SystemMessages are joined into a real system prompt (passed to the CLI via
        ``--system-prompt``). The remaining conversation becomes the user prompt:

        - A single HumanMessage is sent verbatim (no role labels), which keeps the
          prompt identical to what the production API would receive.
        - Multi-turn conversations (the CLI accepts one user turn per query) are
          rendered as ``<conversation_history>`` with ``<user>``, ``<assistant>``,
          ``<tool_call>`` and ``<tool_result>`` elements, followed by an instruction
          for the next step. The explicit structure matters: with plain labelled text,
          models frequently fail to recognize replayed tool results as answers to their
          own calls and request the same tools again (an infinite agent loop).

        Args:
            messages: LangChain messages

        Returns:
            Tuple of (system prompt or None, user prompt)

        Raises:
            ValueError: If the list is empty or contains no usable conversation content
        """
        if not messages:
            raise ValueError("Message list cannot be empty")

        system_parts: List[str] = []
        turns: List[Tuple[BaseMessage, str]] = []

        for i, message in enumerate(messages):
            text = cls.message_text(message, i)
            if not text:
                logger.warning(f"Message {i} has empty content, ignored")
                continue
            if isinstance(message, SystemMessage):
                system_parts.append(text)
            else:
                turns.append((message, text))

        if not turns:
            raise ValueError(
                "No valid message to convert: at least one non-empty, non-system message "
                "is required"
            )

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        if len(turns) == 1 and isinstance(turns[0][0], HumanMessage):
            return system_prompt, turns[0][1]

        history = "\n\n".join(cls._render_turn(msg, i) for i, (msg, _) in enumerate(turns))
        last = turns[-1][0]
        if isinstance(last, (ToolMessage, FunctionMessage)):
            instruction = _AFTER_TOOL_RESULTS
        elif isinstance(last, HumanMessage):
            instruction = "Reply to the latest user message."
        else:
            instruction = "Continue the conversation as the assistant."
        return (
            system_prompt,
            f"<conversation_history>\n{history}\n</conversation_history>\n\n{instruction}",
        )

    @classmethod
    def _render_turn(cls, message: BaseMessage, index: int) -> str:
        text = cls.content_to_text(message.content, index)
        if isinstance(message, HumanMessage):
            return f"<user>\n{text}\n</user>"
        if isinstance(message, AIMessage):
            parts = [text] if text else []
            parts += [
                f'<tool_call id="{call.get("id") or ""}" name="{call["name"]}">'
                f"{json.dumps(call['args'])}</tool_call>"
                for call in message.tool_calls
            ]
            return "<assistant>\n" + "\n".join(parts) + "\n</assistant>"
        if isinstance(message, ToolMessage):
            name = f' name="{message.name}"' if message.name else ""
            return (
                f'<tool_result tool_call_id="{message.tool_call_id}"{name}>\n{text}\n</tool_result>'
            )
        if isinstance(message, FunctionMessage):
            return f'<tool_result name="{message.name}">\n{text}\n</tool_result>'
        role = getattr(message, "role", None) or message.type
        return f"<{role}>\n{text}\n</{role}>"

    @classmethod
    def langchain_to_claude_prompt(cls, messages: List[BaseMessage]) -> str:
        """
        Render all messages (including system messages) as a single labelled transcript.

        Kept for backward compatibility; the chat model itself uses ``split_messages``
        so that system messages become a real system prompt.
        """
        if not messages:
            raise ValueError("Message list cannot be empty")

        parts = []
        for i, message in enumerate(messages):
            text = cls.message_text(message, i)
            if not text:
                logger.warning(f"Message {i} has empty content, ignored")
                continue
            parts.append(cls._label(message, text))

        if not parts:
            raise ValueError("No valid message to convert")
        return "\n\n".join(parts)

    @staticmethod
    def _label(message: BaseMessage, text: str) -> str:
        if isinstance(message, SystemMessage):
            return f"System: {text}"
        if isinstance(message, HumanMessage):
            return f"Human: {text}"
        if isinstance(message, AIMessage):
            return f"Assistant: {text}"
        if isinstance(message, ToolMessage):
            return f"Tool Result [{message.tool_call_id}]: {text}"
        if isinstance(message, FunctionMessage):
            return f"Tool Result: {text}"
        return text

    @staticmethod
    def extract_usage_metadata(result_message: Any) -> Dict[str, Any]:
        """
        Extract response metadata from an SDK ``ResultMessage``.

        Returns:
            Dict with any of: usage, cost_usd, duration_ms, session_id, stop_reason
        """
        metadata: Dict[str, Any] = {}

        try:
            usage = getattr(result_message, "usage", None)
            if usage:
                metadata["usage"] = usage

            cost = getattr(result_message, "total_cost_usd", None)
            if cost is not None:
                metadata["cost_usd"] = float(cost)

            duration = getattr(result_message, "duration_ms", None)
            if duration is not None:
                metadata["duration_ms"] = int(duration)

            session_id = getattr(result_message, "session_id", None)
            if session_id:
                metadata["session_id"] = str(session_id)

            stop_reason = getattr(result_message, "stop_reason", None)
            if stop_reason:
                metadata["stop_reason"] = stop_reason

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error extracting metadata: {e}")

        return metadata

    @staticmethod
    def to_usage_metadata(usage: Optional[Dict[str, Any]]) -> Optional[UsageMetadata]:
        """
        Convert an Anthropic-style usage dict into LangChain ``UsageMetadata``.

        Mirrors ChatAnthropic: ``input_tokens`` includes cache reads and cache writes.
        """
        if not usage:
            return None

        try:
            base_input = int(usage.get("input_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            return None

        input_tokens = base_input + cache_read + cache_creation
        metadata = UsageMetadata(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_token_details={"cache_read": cache_read, "cache_creation": cache_creation},
        )

        details = usage.get("output_tokens_details") or {}
        thinking_tokens = details.get("thinking_tokens") if isinstance(details, dict) else None
        if thinking_tokens:
            metadata["output_token_details"] = {"reasoning": int(thinking_tokens)}

        return metadata
