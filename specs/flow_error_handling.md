# Flow: Error Handling in ClaudeCodeChatModel

## Description
Validates that the adapter handles failures gracefully and gives users actionable
feedback. Failures are simulated with the scripted SDK client, so the flows are
deterministic and need no CLI.

All runtime failures are raised as `ClaudeCodeError`, a subclass of `RuntimeError`, so
existing `except RuntimeError` handlers keep working.

## User Journeys

### SDK Not Installed
1. User creates a `ClaudeCodeChatModel` without `claude-agent-sdk` installed
2. User gets an `ImportError` with `pip install` and CLI install instructions

### CLI Not Installed
1. User invokes the model, and the `claude` executable can't be found
2. User gets a `ClaudeCodeError` with `npm install -g @anthropic-ai/claude-code`

### Process Failure
1. The CLI process fails (authentication, crash, ...)
2. User gets a `ClaudeCodeError` with the exit code and stderr

### Unparseable CLI Output
1. The CLI emits malformed JSON
2. User gets a `ClaudeCodeError` that shows the offending line

### Connection Failure
1. The SDK can't connect to the CLI process
2. User gets a `ClaudeCodeError` saying it could not connect

### Error Result
1. The CLI reports an error result (for example an unknown or retired model)
2. User gets a `ClaudeCodeError` with the CLI's explanation

### Errors While Streaming
1. Some chunks arrive, then the process fails
2. User receives the chunks that arrived before the failure, then the error
3. The CLI process is disconnected

### Recovery
1. A request fails
2. The next request on the same model instance succeeds

### Invalid Input
1. User passes an empty message list, or messages with empty content
2. User gets a `ValueError` before any CLI process starts

### Timeout
Covered in `flow_offline_behavior.md`: `timeout` raises `ClaudeCodeTimeoutError`.

## Success Criteria
- All errors are caught and wrapped with actionable messages
- No silent failures
- The model instance stays usable after an error
- No CLI process is left running after an error
