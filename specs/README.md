# Pragmatic Flow Tests for Claude Code SDK LangChain

## Testing Philosophy

This directory holds pragmatic flow tests written for an LLM-first development approach. These tests:

- **Focus on user journeys** through public APIs only
- **Treat the system as a black box**: private methods and implementation details aren't tested
- **Validate behavior, not implementation**, so they survive refactoring
- **Leave unit-level checks to the LLM**: the model validates logic as it writes code (the LLM is the unit test)

## Two Kinds of Flows

| Kind | Marker | Needs CLI / subscription | Speed |
|------|--------|--------------------------|-------|
| **Offline** | *(none)* | No | ~1 s for the whole suite (51 tests) |
| **Live** | `live` | Yes | ~3 min with `haiku` (24 tests) |

**Offline flows** replace only the SDK client boundary with `fake_sdk.FakeClaudeSDKClient`,
a scripted client that yields **real** SDK message types (`StreamEvent`, `AssistantMessage`,
`ResultMessage`). Everything inside the adapter runs for real. Offline flows cover what a live
model can't make deterministic: exact conversion, stop sequences split across chunks,
error paths, early-exit cleanup and timeouts.

**Live flows** call the real Claude Code CLI and prove the adapter works end to end. They're
skipped automatically when `claude` isn't on PATH, or when `CLAUDE_SKIP_LIVE=1`.

## Test Structure

Each flow has two parts:
1. **Flow description** (`flow_*.md`): documentation of the user journey
2. **Flow test** (`flow_*_test.py`): a pytest implementation that uses only public APIs

## Available Flows

| Flow | Description | Test | Kind |
|------|-------------|------|------|
| Basic chat | `flow_basic_chat.md` | `flow_basic_chat_test.py` | Live |
| LangChain integration | `flow_langchain_integration.md` | `flow_langchain_integration_test.py` | Live |
| Streaming | `flow_streaming.md` | `flow_streaming_test.py` | Live |
| Error handling | `flow_error_handling.md` | `flow_error_handling_test.py` | Offline |
| Adapter behavior | `flow_offline_behavior.md` | `flow_offline_behavior_test.py` | Offline |
| Tool calling | `flow_tool_calling.md` | `flow_tool_calling_offline_test.py` | Offline |
| Tool calling | `flow_tool_calling.md` | `flow_tool_calling_test.py` | Live |

Supporting files: `conftest.py` (the `fake_claude` fixture and live-test skipping),
`fake_sdk.py` (the scripted client and message builders), and `test_helpers.py`
(`get_test_model_name()`).

## Running Tests

```bash
pixi run test-offline                 # offline flows
pixi run test-live                    # live flows
pixi run test                         # everything

pytest specs -m "not live"            # without pixi
pytest specs/flow_streaming_test.py   # one flow
python specs/flow_basic_chat_test.py  # a flow file runs standalone too
```

The live model is chosen with `CLAUDE_TEST_MODEL` (default `haiku`). See `TEST_CONFIGURATION.md`.

## Key Testing Principles

### ✅ DO test
- Public API methods (`invoke()`, `stream()`, `astream()`, `batch()`)
- User-visible behavior and outputs, including metadata
- Error messages users will see
- Integration with LangChain components
- End-to-end flows

### ❌ DON'T test
- Private methods (anything with a `_` prefix)
- Internal state or attributes
- Implementation details
- Code the LLM validates naturally

## Adding New Flow Tests

1. Create the flow description: `specs/flow_[name].md`
2. Create the test: `specs/flow_[name]_test.py`
3. Test ONLY through public APIs
4. Live tests: add `pytestmark = pytest.mark.live` and use `get_test_model_name()`
5. Offline tests: take the `fake_claude` fixture and set `fake_claude.script`

## Test Independence

Each test should:
- Be runnable on its own
- Not depend on other tests
- Clean up its own resources
- Use the model as a black box
