# Flow: Adapter Behavior (Offline)

## Description
Validates the adapter's observable behavior deterministically, using a scripted SDK client
(`fake_sdk.FakeClaudeSDKClient`) that yields real SDK message types. No CLI or subscription
is needed.

## User Journeys

### Getting a Response
1. User invokes the model with a prompt
2. User receives an `AIMessage` with the text, `usage_metadata` (input tokens include cache
   reads and writes, like ChatAnthropic), `response_metadata` (model name, session ID,
   stop reason, cost, duration) and any thinking text in `additional_kwargs`

### Controlling the Request
1. User creates a model with defaults and gets an isolated pure-chat configuration: no
   built-in tools, no MCP servers or account connectors, no filesystem settings, no
   session persistence, one turn
2. User sets `effort`, `builtin_tools`, `max_turns`, `setting_sources`, `mcp_servers`,
   `strict_mcp_config`, `env` or `cwd`, and each value reaches the SDK

### Shaping the Prompt
1. A single human message is sent verbatim
2. System messages become the real system prompt, and they take precedence (with a
   warning) over the constructor `system_prompt`
3. Multi-turn history is sent as a labelled transcript
4. Images are dropped with a warning
5. `temperature` / `max_tokens` log a warning only when a value is set

### Streaming
1. `stream()` and `astream()` yield token-level chunks that aggregate to the full message
2. When the CLI doesn't send partial events, the complete blocks are used instead
3. Subagent output (`parent_tool_use_id`) never leaks into the reply
4. Callbacks receive every token
5. `prompt | model | StrOutputParser()` streams strings asynchronously

### Stopping Early
1. A stop sequence truncates the output, even when the sequence is split across
   chunks, sets `stop_reason="stop_sequence"`, and interrupts the CLI
2. Breaking out of a stream interrupts and disconnects the CLI
3. A completed request disconnects without an interrupt
4. `timeout` raises `ClaudeCodeTimeoutError` (also a `TimeoutError`) and cleans up

### Runtime Environments
1. Sync `invoke()` works while an event loop is running (Jupyter)
2. `batch()` runs independent requests

## Success Criteria
- Every journey passes deterministically (the suite is run repeatedly to catch races)
- Only public APIs are used, plus the LangSmith tracing params contract
