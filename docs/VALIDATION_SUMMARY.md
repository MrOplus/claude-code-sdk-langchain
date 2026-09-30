# Validation Summary

## v0.3.0 - 2026-09-30 (tool calling)

Same environment as v0.2.0, plus langchain 1.4.3 for agent tests.

| Suite | Tests | Result |
|-------|-------|--------|
| Offline flows | 51 | ✅ 51 passed (14 new tool-calling flows) |
| Live flows | 24 | ✅ 24 passed (6 new tool-calling flows) |

**Verified live**: native tool calls returned without execution, parallel calls, text answers
when no tool is needed, a manual tool loop that finishes without repeated calls, a forced
named tool, `with_structured_output(Pydantic)`, and `create_agent` with sequential dependent
tools (`find_city` → `city_weather`).

**Issue found during development**: replaying tool results as labelled plain text made Haiku
repeat the same calls in 10/10 replays that included a system prompt (`create_agent` looped
until its recursion limit). Structured `<conversation_history>` replay with a closing
instruction brought this to 0/10 (haiku) and 0/5 (sonnet), and 3/3 sequential agent runs and
3/3 parallel agent runs completed.

## v0.2.0 - 2026-09-30

**Environment**: Windows 11, Python 3.11.13, langchain-core 1.6.6, claude-agent-sdk 0.2.162,
Claude Code CLI 2.1.285, live model `haiku` (resolved to `claude-haiku-4-5-20251001`).

### Results

| Suite | Tests | Result |
|-------|-------|--------|
| Offline flows (scripted SDK client) | 37 | ✅ 37 passed, stable across 10 consecutive runs |
| Live flows (real Claude Code CLI) | 18 | ✅ 18 passed |
| Offline flows inside the pixi environment | 37 | ✅ 37 passed |

### What Was Verified Live

- `invoke` / `ainvoke` return an `AIMessage` with `usage_metadata` and the resolved `model_name`
- System messages are followed (they are sent as the real system prompt)
- Multi-turn history is used by the model
- Stop sequences truncate output and report `stop_reason="stop_sequence"`
- `stream` / `astream` deliver **token-level** chunks that arrive over time
- Streamed chunks aggregate into a full message with usage metadata
- LCEL chains, output parsers (sync and async streaming), batch and multi-step chains
- Breaking out of a stream leaves the model usable, and the CLI process is interrupted and reaped
- Default requests are isolated: no built-in or MCP tools, under 1,500 input tokens

### Issues Found and Fixed in This Release

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | Critical | The default model `claude-sonnet-4-20250514` is no longer accessible; every default request failed | Default changed to the `sonnet` alias |
| 2 | Critical | `claude-code-sdk` is superseded by `claude-agent-sdk` | Migrated; the options, errors and client lifecycle were updated |
| 3 | High | Abandoned requests (early `break`, stop sequence, timeout) left the CLI process generating in the background | Requests now use `ClaudeSDKClient` with explicit `interrupt()` + `disconnect()`. Pending cancellation is cleared during cleanup so the SDK's shutdown can finish |
| 4 | High | Streaming yielded whole text blocks, not tokens | Uses partial stream events (`text_delta`) with a fallback to full blocks |
| 5 | High | The Claude Code default tools, CLAUDE.md files and user hooks leaked into "chat" requests | Tools are disabled and `setting_sources=[]` by default |
| 5b | High | claude.ai account connectors (Gmail, Drive, Docs, ...) were attached to every request: 8 MCP tools, about 2,900 extra input tokens (3,377 vs. 444), and answers that mention those tools | `strict_mcp_config=True` by default; `mcp_servers` attaches servers explicitly. A live regression test asserts input tokens stay under 1,500 |
| 6 | Medium | System messages were inlined as `System:` text | Sent as the real system prompt |
| 7 | Medium | Every request wrote a session transcript to `~/.claude` | `--no-session-persistence` by default (`persist_session=True` to opt in) |
| 8 | Medium | No `usage_metadata`, so LangChain token accounting was empty | `UsageMetadata` is populated the same way ChatAnthropic does it |
| 9 | Medium | Stop sequences were ignored | Emulated client-side, including sequences split across chunks |
| 10 | Low | Warnings fired for the default `temperature=0.7` / `max_tokens=2000` sentinels | Defaults are now `None`; warnings fire only when a value is set |
| 11 | Low | Tokens could be dispatched twice to callbacks | langchain-core 1.x dispatches centrally; the manual dispatch was removed |

### Known Limitations

See the README's *Limitations* section: `temperature` and `max_tokens` are not supported,
images are dropped, there is no native `bind_tools`, and multi-turn history is sent as a
transcript.

`CLAUDE_CODE_MAX_OUTPUT_TOKENS` was evaluated as a way to implement `max_tokens`. When
the limit is exceeded the CLI **fails the request** instead of truncating, which is
incompatible with API semantics, so it was not adopted.

---

## v0.1.0 - 2025-09-30 (historical)

The original validation found 19 logic bugs across two review sessions. The critical ones
were silent `temperature`/`max_tokens` handling, conflicting system prompts and missing
multimodal handling. They were fixed with warnings and documentation. 16/16 live flow
tests passed with `claude-code-sdk` 0.0.23.
