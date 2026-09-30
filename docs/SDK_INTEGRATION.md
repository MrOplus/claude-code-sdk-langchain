# Claude Agent SDK Integration Notes

How `ClaudeCodeChatModel` uses the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/)
(`claude-agent-sdk`, the successor of `claude-code-sdk`). For the complete SDK reference, see
the official docs: https://docs.claude.com/en/docs/agent-sdk/python

## Request Lifecycle

Each LangChain call (`invoke`, `stream`, ...) is one short-lived SDK session:

```python
client = ClaudeSDKClient(options=options)
await client.connect()                 # spawns the `claude` CLI process
await client.query(prompt)             # sends one user turn
async for message in client.receive_response():
    ...                                # StreamEvent / AssistantMessage / ResultMessage
# on early exit (break, stop sequence, timeout):
await client.interrupt()               # stop generating
await client.disconnect()              # always: close pipes, reap the process
```

`ClaudeSDKClient` is used instead of the one-shot `query()` helper. `query()` does not
close its inner generator when it is closed early, which leaves the CLI process
running (and using quota) until it finishes on its own.

## Options Mapping

| Adapter field | `ClaudeAgentOptions` | Default | Notes |
|---------------|----------------------|---------|-------|
| `model` | `model` | `"sonnet"` | Aliases resolve to the latest model |
| system messages / `system_prompt` | `system_prompt` | `None` → empty system prompt | SystemMessage wins over the constructor value |
| `builtin_tools` | `tools` | `[]` | `[]` disables all built-in tools (pure chat) |
| `allowed_tools` | `allowed_tools` | `[]` | Auto-approved tools |
| `max_turns` | `max_turns` | `1` | Raise it when enabling tools |
| `setting_sources` | `setting_sources` | `[]` | `[]` = SDK isolation mode, `None` = load user/project/local settings |
| `mcp_servers` | `mcp_servers` | `{}` | MCP servers to attach explicitly |
| `strict_mcp_config` | `strict_mcp_config` | `True` | Ignore MCP servers from settings **and claude.ai account connectors** |
| `effort` | `effort` | `None` | `low`, `medium`, `high`, `xhigh`, `max` |
| `permission_mode` | `permission_mode` | `None` | |
| `cwd`, `env`, `cli_path` | same | | |
| `persist_session=False` | `extra_args={"no-session-persistence": None}` | | Don't write transcripts to `~/.claude` |
| (always) | `include_partial_messages=True` | | Enables token-level `StreamEvent`s |
| `temperature`, `max_tokens` | *(none)* | `None` | Not supported by the CLI (see the investigation doc) |

## Message Handling

**Input** (`MessageConverter.split_messages`):
- `SystemMessage` content → `system_prompt`
- a single `HumanMessage` → sent verbatim
- multi-turn history → a `<conversation_history>` block with `<user>`, `<assistant>`, `<tool_call>` and `<tool_result>` elements plus a next-step instruction (the CLI accepts one user turn per query)
- image and other non-text parts → dropped, with a warning

**Output** (`ClaudeCodeChatModel._iter_chunks`):

| SDK message | Becomes |
|-------------|---------|
| `StreamEvent` with `content_block_delta` / `text_delta` | `AIMessageChunk(content=text)` |
| `StreamEvent` with `thinking_delta` | `AIMessageChunk(additional_kwargs={"thinking": ...})` |
| `AssistantMessage` | Model name; its text/thinking blocks are used only if no deltas were streamed for them; `mcp__langchain__*` `ToolUseBlock`s become tool call chunks |
| `ResultMessage` | Final chunk with `usage_metadata` and `response_metadata`; `is_error` → `ClaudeCodeError` |
| Anything with `parent_tool_use_id` | Ignored (subagent output) |

## Tool Calling

`bind_tools()` converts tools to OpenAI-style schemas (`convert_to_openai_tool`) and binds
them as `tools` / `tool_choice` kwargs. For each request that has tools:

1. The schemas are served to Claude as native tools by an **in-process MCP server**
   (`create_sdk_mcp_server`, server name `langchain`, so tools appear as
   `mcp__langchain__<name>`).
2. A **PreToolUse hook** matching `^mcp__langchain__` returns
   `permissionDecision: "defer"`. The CLI stops the run (`stop_reason: "tool_deferred"`)
   without executing anything; the MCP handlers are never called.
3. Every `ToolUseBlock` for a `mcp__langchain__*` tool in the `AssistantMessage` becomes a
   LangChain `tool_call_chunk`, so parallel calls are all reported. The response's
   `stop_reason` is normalized to `"tool_use"`.
4. On the next request, the history (including `AIMessage.tool_calls` and `ToolMessage`s) is
   replayed inside `<conversation_history>` with `<tool_call>` / `<tool_result>` elements,
   followed by an instruction not to repeat calls whose results are already present.

`tool_choice`: a named tool is enforced by exposing only that tool. `"any"` / `"required"` is
enforced by a system-prompt instruction. `"none"` exposes no tools.

Built-in Claude Code tools (`builtin_tools`) are not deferred. They run inside the CLI and
are not reported as LangChain tool calls.

### Why structured replay

With labelled plain text (`Human:` / `Assistant: [Tool call] ...` / `Tool Result: ...`),
Haiku repeated the same tool calls in **10/10** replays that included a system prompt,
which put `create_agent` into an infinite loop. With the XML-structured history and the
closing instruction: **0/10** (haiku), **0/5** (sonnet), and sequential dependent tool calls
still work.

## Error Mapping

| SDK exception | Raised as |
|---------------|-----------|
| `CLINotFoundError` | `ClaudeCodeError` with install instructions |
| `ProcessError` / `ResultError` | `ClaudeCodeError` with the exit code and stderr |
| `CLIJSONDecodeError` | `ClaudeCodeError` with the offending line |
| `CLIConnectionError` | `ClaudeCodeError` |
| `timeout` exceeded | `ClaudeCodeTimeoutError` (subclass of `ClaudeCodeError` and `TimeoutError`) |

`ClaudeCodeError` subclasses `RuntimeError`, so `except RuntimeError` handlers from 0.1.x keep working.

## Concurrency Notes

- **anyio isolation**: the SDK runs an anyio task group per client. `_astream` consumes
  the SDK in a dedicated asyncio task and forwards chunks through a queue, so LangChain
  wrappers (such as `StrOutputParser` in `chain.astream`) never enter or exit anyio cancel
  scopes from a different task.
- **Cancellation**: a raw asyncio cancellation (early `break`, `asyncio.timeout`) would
  abort the SDK's shutdown at its first `await` and orphan the CLI process. The client
  wrapper calls `task.uncancel()` before cleaning up, so `interrupt()` and `disconnect()`
  can complete. The in-flight `CancelledError` still propagates.
- **Sync API**: `_generate` runs the async path with `asyncio.run`, or in a worker thread
  if a loop is already running (Jupyter). `_stream` runs `_astream` on a private loop in a
  worker thread and cancels it when iteration stops.
