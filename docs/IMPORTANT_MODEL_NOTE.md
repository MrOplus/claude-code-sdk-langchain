# Model Selection Note

## Default Model

The adapter's default model is the **`sonnet` alias**:

```python
ClaudeCodeChatModel()                 # same as model="sonnet"
```

The Claude Code CLI resolves aliases to the latest model of that family. The resolved
ID is reported in `response.response_metadata["model_name"]` (for example
`claude-haiku-4-5-20251001` for `haiku`).

## History

Versions up to 0.1.0 hard-coded `claude-sonnet-4-20250514` as the default, and this note
said never to change it. On 2026-09-30 a live check showed that ID is **no longer
accessible** through the CLI:

```
Claude Code returned an error result: There's an issue with the selected model
(claude-sonnet-4-20250514). It may not exist or you may not have access to it.
```

Every request made with the old default therefore failed. Version 0.2.0 switched to the
`sonnet` alias so the default keeps working as models are retired.

## Guidance

- **Prefer aliases** (`haiku`, `sonnet`, `opus`) for prototyping. They never expire.
- **Pin a full model ID** only when you need exact reproducibility, and expect to update
  it when that model is retired.
- **Tests** use `CLAUDE_TEST_MODEL` (default `haiku`), which is fast and light on quota.
- If a request fails with "There's an issue with the selected model", the ID has been
  retired or your plan doesn't include it. Switch to an alias.
