"""
LangChain chat model backed by the Claude Agent SDK (Claude Code CLI).
"""

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import queue
import threading
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from .message_converter import MessageConverter

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        CLIConnectionError,
        CLIJSONDecodeError,
        CLINotFoundError,
        HookMatcher,
        ProcessError,
        ResultMessage,
        SdkMcpTool,
        StreamEvent,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        create_sdk_mcp_server,
    )

    CLAUDE_CODE_AVAILABLE = True
except ImportError as e:  # pragma: no cover - exercised via monkeypatching in tests
    CLAUDE_CODE_AVAILABLE = False
    logger.error(f"claude-agent-sdk not installed: {e}")

DEFAULT_MODEL = "sonnet"
"""Default model alias. The CLI resolves it to the latest Sonnet model."""

_INTERRUPT_TIMEOUT = 5.0
"""Seconds to wait for the CLI to acknowledge an interrupt on early exit."""

_TOOL_SERVER = "langchain"
"""In-process MCP server name under which bound LangChain tools are exposed."""

_TOOL_PREFIX = f"mcp__{_TOOL_SERVER}__"
"""Prefix the CLI adds to the names of tools served by ``_TOOL_SERVER``."""

_T = TypeVar("_T")


class ClaudeCodeError(RuntimeError):
    """Raised when the Claude Code CLI fails or reports an error result."""


class ClaudeCodeTimeoutError(ClaudeCodeError, TimeoutError):
    """Raised when a request exceeds the configured ``timeout``."""


class _StopSequenceFilter:
    """
    Emulates API-side stop sequences on a stream of text.

    Text that could still be the beginning of a stop sequence is held back until
    it is disambiguated, so streamed output never contains a stop sequence.
    """

    def __init__(self, stop: Optional[List[str]]):
        self.stop = [s for s in (stop or []) if s]
        self._holdback = max((len(s) for s in self.stop), default=1) - 1
        self._buffer = ""
        self.triggered = False

    def feed(self, text: str) -> str:
        """Add text; return the part that is safe to emit."""
        if not self.stop:
            return text
        if self.triggered:
            return ""

        self._buffer += text
        positions = [p for p in (self._buffer.find(s) for s in self.stop) if p != -1]
        if positions:
            self.triggered = True
            out, self._buffer = self._buffer[: min(positions)], ""
            return out

        if self._holdback == 0:
            out, self._buffer = self._buffer, ""
            return out
        out = self._buffer[: -self._holdback]
        self._buffer = self._buffer[-self._holdback :]
        return out

    def flush(self) -> str:
        """Return any held-back text at end of stream."""
        out, self._buffer = ("" if self.triggered else self._buffer), ""
        return out


class _StreamError:
    """Wraps an exception raised by the background stream consumer."""

    def __init__(self, exc: BaseException):
        self.exc = exc


_DONE = object()


class ClaudeCodeChatModel(BaseChatModel):
    """
    LangChain chat model that uses Claude through your Claude Code subscription.

    Requests are executed by the Claude Code CLI via the Claude Agent SDK, so
    prototyping incurs no per-token API charges. By default the model behaves as a
    plain chat model: built-in tools and MCP servers (including claude.ai account
    connectors) are disabled, filesystem settings (CLAUDE.md, hooks, ...) are not
    loaded, and sessions are not persisted to disk.

    Limitations (compared with ChatAnthropic):
        - ``temperature``: accepted for compatibility, not supported by the CLI
        - ``max_tokens``: accepted for compatibility, not supported by the CLI
        - Images / multimodal input: ignored with a warning
        - ``stop``: emulated client-side (output is truncated at the stop sequence)

    Example:
        .. code-block:: python

            from claude_code_langchain import ClaudeCodeChatModel

            model = ClaudeCodeChatModel(model="sonnet")
            print(model.invoke("Hello!").content)

            for chunk in model.stream("Tell me a story"):
                print(chunk.content, end="", flush=True)

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [("system", "You are an assistant"), ("human", "{input}")]
            )
            chain = prompt | model
            chain.invoke({"input": "What is LangChain?"})
    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model alias (``sonnet``, ``opus``, ``haiku``) or full model ID."""

    temperature: Optional[float] = None
    """NOT SUPPORTED by the Claude Code CLI. Accepted for API compatibility only."""

    max_tokens: Optional[int] = None
    """NOT SUPPORTED by the Claude Code CLI. Accepted for API compatibility only."""

    stop_sequences: Optional[List[str]] = Field(default=None, alias="stop")
    """Default stop sequences, emulated client-side by truncating the output."""

    system_prompt: Optional[str] = None
    """Default system prompt. A SystemMessage in the input takes precedence."""

    effort: Optional[str] = None
    """Reasoning effort level: ``low``, ``medium``, ``high``, ``xhigh`` or ``max``."""

    timeout: Optional[float] = None
    """Maximum seconds per request. ``None`` means no limit."""

    builtin_tools: List[str] = Field(default_factory=list)
    """Claude Code built-in tools to make available (e.g. ``["WebSearch"]``).
    Empty by default, which makes the model a pure chat model."""

    allowed_tools: List[str] = Field(default_factory=list)
    """Tools that may run without a permission prompt."""

    max_turns: Optional[int] = 1
    """Maximum agent turns. Increase this when enabling ``builtin_tools``."""

    permission_mode: Optional[str] = None
    """Claude Code permission mode: default, acceptEdits, plan, bypassPermissions."""

    cwd: Optional[str] = None
    """Working directory for the Claude Code process."""

    setting_sources: Optional[List[str]] = Field(default_factory=list)
    """Filesystem settings to load (``user``, ``project``, ``local``). Empty by default
    for reproducible, isolated behavior; ``None`` loads everything like the CLI does."""

    mcp_servers: Dict[str, Any] = Field(default_factory=dict)
    """MCP servers to attach, in ``ClaudeAgentOptions.mcp_servers`` format."""

    strict_mcp_config: bool = True
    """Use only ``mcp_servers``. When ``False``, the CLI also attaches MCP servers from
    your settings and claude.ai account connectors, which adds their tool definitions
    (often thousands of tokens) to every request."""

    persist_session: bool = False
    """Whether the CLI should save session transcripts to disk."""

    env: Dict[str, str] = Field(default_factory=dict)
    """Extra environment variables for the Claude Code process."""

    cli_path: Optional[str] = None
    """Explicit path to the ``claude`` executable. Auto-detected when ``None``."""

    def __init__(self, **kwargs: Any):
        if not CLAUDE_CODE_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is not installed. "
                "Install it with: pip install claude-agent-sdk\n"
                "and make sure the Claude Code CLI is installed: "
                "npm install -g @anthropic-ai/claude-code"
            )

        super().__init__(**kwargs)

        if self.temperature is not None:
            logger.warning(
                f"temperature={self.temperature} is NOT SUPPORTED by the Claude Code CLI and "
                "will be ignored. Use ChatAnthropic for temperature control."
            )
        if self.max_tokens is not None:
            logger.warning(
                f"max_tokens={self.max_tokens} is NOT SUPPORTED by the Claude Code CLI and "
                "will be ignored. Use ChatAnthropic for token limit control."
            )

    # ------------------------------------------------------------------ options

    def _resolve_system_prompt(self, message_system_prompt: Optional[str]) -> Optional[str]:
        if message_system_prompt is not None:
            if self.system_prompt:
                logger.warning(
                    "Both constructor system_prompt and SystemMessage detected. "
                    "Using SystemMessage from messages (takes precedence). "
                    "Constructor system_prompt will be ignored."
                )
            return message_system_prompt
        return self.system_prompt

    def _build_options(
        self,
        system_prompt: Optional[str],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
    ) -> "ClaudeAgentOptions":
        extra_args: Dict[str, Optional[str]] = {}
        if not self.persist_session:
            extra_args["no-session-persistence"] = None

        mcp_servers = dict(self.mcp_servers)
        allowed_tools = list(self.allowed_tools)
        hooks = None

        if tools and tool_choice != "none":
            forced = [t for t in tools if t["function"]["name"] == tool_choice]
            if forced:
                tools = forced  # a forced tool is the only one Claude can call
            mcp_servers[_TOOL_SERVER] = _deferred_tool_server(tools)
            allowed_tools += [_TOOL_PREFIX + t["function"]["name"] for t in tools]
            hooks = {
                "PreToolUse": [
                    HookMatcher(matcher=f"^{_TOOL_PREFIX}", hooks=[_defer_langchain_tool])
                ]
            }
            instruction = _tool_choice_instruction(tool_choice)
            if instruction:
                system_prompt = (
                    f"{system_prompt}\n\n{instruction}" if system_prompt else instruction
                )

        return ClaudeAgentOptions(
            model=self.model_name,
            system_prompt=system_prompt,
            tools=list(self.builtin_tools),
            allowed_tools=allowed_tools,
            max_turns=self.max_turns,
            permission_mode=self.permission_mode,  # type: ignore[arg-type]
            cwd=self.cwd,
            setting_sources=self.setting_sources,  # type: ignore[arg-type]
            mcp_servers=mcp_servers,
            strict_mcp_config=self.strict_mcp_config,
            hooks=hooks,  # type: ignore[arg-type]
            effort=self.effort,  # type: ignore[arg-type]
            env=dict(self.env),
            cli_path=self.cli_path,
            extra_args=extra_args,
            include_partial_messages=True,
            stderr=lambda line: logger.debug(f"claude stderr: {line}"),
        )

    def _prepare(
        self, messages: List[BaseMessage], stop: Optional[List[str]], kwargs: Dict[str, Any]
    ) -> Tuple[str, "ClaudeAgentOptions", Optional[List[str]]]:
        kwargs = dict(kwargs)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        kwargs.pop("ls_structured_output_format", None)  # tracing hint from with_structured_output
        if kwargs:
            logger.warning(
                f"Additional parameters {list(kwargs.keys())} are not supported and will be "
                "ignored."
            )
        message_system, prompt = MessageConverter.split_messages(messages)
        options = self._build_options(
            self._resolve_system_prompt(message_system), tools, tool_choice
        )
        return prompt, options, stop if stop is not None else self.stop_sequences

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, bool, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        Bind tools so the model can request tool calls.

        Tools are exposed to Claude as native tools. When Claude calls one, the
        request stops and the call is returned in ``AIMessage.tool_calls``; nothing is
        executed. Run the tool yourself (or with ``ToolNode`` / an agent) and pass the
        result back as a ``ToolMessage``.

        Args:
            tools: LangChain tools, functions, Pydantic models or OpenAI-style tool dicts.
            tool_choice: ``None``/``"auto"`` (model decides), ``"any"``/``"required"``/
                ``True`` (must call a tool), a tool name, or ``"none"``. Forcing a tool is
                done by instruction, so it is best-effort.
        """
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        if isinstance(tool_choice, dict):
            tool_choice = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        return super().bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    # ---------------------------------------------------------------- core loop

    async def _sdk_messages(
        self, prompt: str, options: "ClaudeAgentOptions"
    ) -> AsyncGenerator[Any, None]:
        """
        Yield SDK messages for a single request, owning the CLI process lifecycle.

        If the consumer stops early (stop sequence, ``break``, cancellation), the
        request is interrupted so the CLI stops generating instead of running on in
        the background, and the process is always shut down before returning.
        """
        client = ClaudeSDKClient(options=options)
        connected = False
        finished = False
        try:
            await client.connect()
            connected = True
            await client.query(prompt)
            # receive_response() is an async generator, though annotated as AsyncIterator
            responses = client.receive_response()
            async with contextlib.aclosing(responses):  # type: ignore[type-var]
                async for message in responses:
                    yield message
            finished = True
        finally:
            if connected:
                # A pending asyncio cancellation (early break, timeout) would abort the
                # SDK's shutdown at its first await and orphan the CLI process. Clear
                # the pending-cancel state so cleanup can run; the CancelledError that
                # brought us here is already propagating and is not affected.
                task = asyncio.current_task()
                while task is not None and task.cancelling():
                    task.uncancel()
                if not finished:
                    with contextlib.suppress(Exception):
                        async with asyncio.timeout(_INTERRUPT_TIMEOUT):
                            await client.interrupt()
                with contextlib.suppress(Exception):
                    await client.disconnect()

    async def _iter_chunks(
        self, prompt: str, options: "ClaudeAgentOptions", stop: Optional[List[str]]
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Run one query and yield LangChain chunks.

        Text is streamed token-by-token from partial stream events. If the CLI does not
        emit partial events for a block, the complete block from the AssistantMessage
        is used instead. Calls to bound LangChain tools are emitted as tool call chunks
        (the CLI defers them instead of executing). The final chunk carries usage and
        response metadata.
        """
        stop_filter = _StopSequenceFilter(stop)
        tool_calls_emitted = 0
        text_deltas = 0
        thinking_deltas = 0
        model_name = self.model_name
        response_metadata: Dict[str, Any] = {}
        result: Optional[Any] = None

        def text_chunk(text: str) -> Optional[ChatGenerationChunk]:
            emitted = stop_filter.feed(text)
            return ChatGenerationChunk(message=AIMessageChunk(content=emitted)) if emitted else None

        def thinking_chunk(thinking: str) -> ChatGenerationChunk:
            return ChatGenerationChunk(
                message=AIMessageChunk(content="", additional_kwargs={"thinking": thinking})
            )

        try:
            async with contextlib.aclosing(self._sdk_messages(prompt, options)) as stream:
                async for message in stream:
                    if isinstance(message, StreamEvent):
                        if message.parent_tool_use_id:
                            continue
                        event = message.event or {}
                        if event.get("type") != "content_block_delta":
                            continue
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            text_deltas += 1
                            chunk = text_chunk(delta.get("text", ""))
                            if chunk:
                                yield chunk
                        elif delta.get("type") == "thinking_delta":
                            thinking_deltas += 1
                            yield thinking_chunk(delta.get("thinking", ""))

                    elif isinstance(message, AssistantMessage):
                        if message.parent_tool_use_id:
                            continue
                        model_name = message.model or model_name
                        has_text = any(isinstance(b, TextBlock) for b in message.content)
                        has_thinking = any(isinstance(b, ThinkingBlock) for b in message.content)
                        for block in message.content:
                            if isinstance(block, TextBlock) and text_deltas == 0:
                                chunk = text_chunk(block.text)
                                if chunk:
                                    yield chunk
                            elif isinstance(block, ThinkingBlock) and thinking_deltas == 0:
                                yield thinking_chunk(block.thinking)
                            elif isinstance(block, ToolUseBlock) and block.name.startswith(
                                _TOOL_PREFIX
                            ):
                                yield ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        content="",
                                        tool_call_chunks=[
                                            tool_call_chunk(
                                                name=block.name[len(_TOOL_PREFIX) :],
                                                args=json.dumps(block.input),
                                                id=block.id,
                                                index=tool_calls_emitted,
                                            )
                                        ],
                                    )
                                )
                                tool_calls_emitted += 1
                        if has_text:
                            text_deltas = 0
                        if has_thinking:
                            thinking_deltas = 0

                    elif isinstance(message, ResultMessage):
                        if message.is_error:
                            detail = message.result or ", ".join(message.errors or [])
                            raise ClaudeCodeError(f"Claude Code error: {detail or message.subtype}")
                        result = message

                    if stop_filter.triggered:
                        break

        except ClaudeCodeError:
            raise
        except CLINotFoundError as e:
            raise ClaudeCodeError(
                f"Claude Code CLI not found: {e}\n"
                "Please install: npm install -g @anthropic-ai/claude-code"
            ) from e
        except CLIJSONDecodeError as e:
            raise ClaudeCodeError(
                f"Failed to parse Claude Code response: {e}\nInvalid line: {e.line}"
            ) from e
        except ProcessError as e:
            raise ClaudeCodeError(
                f"Claude Code process error (exit code {e.exit_code}): {e}\nStderr: {e.stderr}"
            ) from e
        except CLIConnectionError as e:
            raise ClaudeCodeError(f"Could not connect to Claude Code: {e}") from e

        tail = stop_filter.flush()
        if tail:
            yield ChatGenerationChunk(message=AIMessageChunk(content=tail))

        if result is not None:
            response_metadata.update(MessageConverter.extract_usage_metadata(result))
        if stop_filter.triggered:
            response_metadata["stop_reason"] = "stop_sequence"
        elif tool_calls_emitted:
            response_metadata["stop_reason"] = "tool_use"  # CLI reports "tool_deferred"
        response_metadata["model_name"] = model_name

        usage = MessageConverter.to_usage_metadata(getattr(result, "usage", None))
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                additional_kwargs={"model": model_name},
                response_metadata=response_metadata,
                usage_metadata=usage,
            )
        )

    @contextlib.asynccontextmanager
    async def _deadline(self) -> AsyncIterator[None]:
        if self.timeout is None:
            yield
            return
        try:
            async with asyncio.timeout(self.timeout):
                yield
        except TimeoutError as e:
            raise ClaudeCodeTimeoutError(
                f"Claude Code request timed out after {self.timeout} seconds"
            ) from e

    # ------------------------------------------------------------ LangChain API

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt, options, stop = self._prepare(messages, stop, kwargs)

        aggregate: Optional[AIMessageChunk] = None
        async with self._deadline():
            async for chunk in self._iter_chunks(prompt, options, stop):
                msg = chunk.message
                assert isinstance(msg, AIMessageChunk)
                aggregate = msg if aggregate is None else aggregate + msg

        if aggregate is None:
            aggregate = AIMessageChunk(content="")

        message = AIMessage(
            content=aggregate.content,
            additional_kwargs=aggregate.additional_kwargs,
            response_metadata=aggregate.response_metadata,
            usage_metadata=aggregate.usage_metadata,
            tool_calls=aggregate.tool_calls,
            invalid_tool_calls=aggregate.invalid_tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return _run_sync(self._agenerate(messages, stop, None, **kwargs))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Stream chunks asynchronously.

        Token callbacks are dispatched by LangChain itself for every yielded chunk.

        The SDK stream is consumed in a dedicated task and forwarded through a queue.
        This isolates the SDK's anyio task group from the caller's task, which avoids
        "cancel scope in a different task" errors when LangChain wraps the stream
        (e.g. ``prompt | model | StrOutputParser()``).
        """
        prompt, options, stop = self._prepare(messages, stop, kwargs)
        chunk_queue: asyncio.Queue = asyncio.Queue()

        async def consume() -> None:
            try:
                async with self._deadline():
                    async for chunk in self._iter_chunks(prompt, options, stop):
                        await chunk_queue.put(chunk)
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001 - forwarded to the caller
                await chunk_queue.put(_StreamError(e))
                return
            await chunk_queue.put(_DONE)

        consumer = asyncio.create_task(consume())
        try:
            while True:
                item = await chunk_queue.get()
                if item is _DONE:
                    break
                if isinstance(item, _StreamError):
                    raise item.exc
                yield item
        finally:
            if not consumer.done():
                consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream chunks synchronously.

        Runs ``_astream`` on a private event loop in a worker thread. Stopping
        iteration early cancels the underlying request instead of letting it run on.
        """
        chunk_queue: queue.Queue = queue.Queue()
        loop = asyncio.new_event_loop()
        cancelled = threading.Event()

        async def pump() -> None:
            try:
                stream = self._astream(messages, stop, None, **kwargs)
                async with contextlib.aclosing(stream):  # type: ignore[type-var]
                    async for chunk in stream:
                        if cancelled.is_set():
                            return
                        chunk_queue.put(chunk)
            except asyncio.CancelledError:
                return
            except BaseException as e:  # noqa: BLE001 - forwarded to the caller
                chunk_queue.put(_StreamError(e))
            finally:
                chunk_queue.put(_DONE)

        # Created before the loop runs, so the main thread can cancel it safely.
        pump_task = loop.create_task(pump())

        def worker() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(pump_task)
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    # Give subprocess transports a chance to close before the loop does.
                    loop.run_until_complete(asyncio.sleep(0))
                finally:
                    loop.close()

        thread = threading.Thread(target=worker, name="claude-code-stream", daemon=True)
        thread.start()
        try:
            while True:
                item = chunk_queue.get()
                if item is _DONE:
                    break
                if isinstance(item, _StreamError):
                    raise item.exc
                yield item
        finally:
            cancelled.set()
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(pump_task.cancel)
            thread.join()

    # --------------------------------------------------------------- metadata

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "effort": self.effort,
            "permission_mode": self.permission_mode,
            "builtin_tools": self.builtin_tools,
            "max_turns": self.max_turns,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> LangSmithParams:
        params = LangSmithParams(
            ls_provider="anthropic",
            ls_model_name=self.model_name,
            ls_model_type="chat",
        )
        ls_stop = stop or self.stop_sequences
        if ls_stop:
            params["ls_stop"] = ls_stop
        return params


async def _defer_langchain_tool(input_data: Any, tool_use_id: Any, context: Any) -> Any:
    """PreToolUse hook: stop the run and hand the tool call back to the caller."""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}


def _deferred_tool_server(tools: List[Dict[str, Any]]) -> Any:
    """
    Expose OpenAI-format tool schemas to Claude as native tools.

    The handlers never run: the PreToolUse hook defers every call, so tool execution
    stays with the LangChain application.
    """

    async def not_executed(args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "content": [{"type": "text", "text": "Tool execution is handled by LangChain."}],
            "is_error": True,
        }

    sdk_tools = [
        SdkMcpTool(
            name=t["function"]["name"],
            description=t["function"].get("description") or t["function"]["name"],
            input_schema=t["function"].get("parameters") or {"type": "object", "properties": {}},
            handler=not_executed,
        )
        for t in tools
    ]
    return create_sdk_mcp_server(name=_TOOL_SERVER, tools=sdk_tools)


def _tool_choice_instruction(tool_choice: Any) -> Optional[str]:
    """Best-effort emulation of ``tool_choice`` (the CLI has no native equivalent)."""
    if tool_choice in (None, "auto", False, "none"):
        return None
    if tool_choice in ("any", "required", True):
        return "You must respond by calling one or more of the available tools."
    return f"You must respond by calling the `{tool_choice}` tool."


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine from sync code, even when an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()
