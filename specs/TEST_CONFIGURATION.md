# Test Configuration Guide

## Model Selection for Live Tests

Live flow tests make real requests through the Claude Code CLI, which takes time and uses
subscription quota. Choose the model with the `CLAUDE_TEST_MODEL` environment variable.

```bash
CLAUDE_TEST_MODEL=haiku pytest specs -m live     # fast (default)
CLAUDE_TEST_MODEL=sonnet pytest specs -m live    # more thorough
CLAUDE_TEST_MODEL=opus pytest specs -m live      # most capable, slowest
```

`pixi.toml` sets `CLAUDE_TEST_MODEL=haiku` for `pixi run` / `pixi shell`. Override it per run:

```bash
CLAUDE_TEST_MODEL=sonnet pixi run test-live
```

### Aliases vs. Full IDs

Prefer the aliases `haiku`, `sonnet` and `opus`, which resolve to the latest model of each
family. Full IDs (for example `claude-haiku-4-5-20251001`) work until that model is retired.
After that, requests fail with "There's an issue with the selected model".

The resolved model is available as `response.response_metadata["model_name"]`.

## Skipping Live Tests

Live tests are skipped automatically when the `claude` executable isn't on PATH. To skip
them explicitly (for example in CI without a subscription):

```bash
CLAUDE_SKIP_LIVE=1 pytest specs
# or
pytest specs -m "not live"
```

## Implementation

```python
import pytest
from claude_code_langchain import ClaudeCodeChatModel
from .test_helpers import get_test_model_name

pytestmark = pytest.mark.live

def test_something():
    model = ClaudeCodeChatModel(model=get_test_model_name())
```

`get_test_model_name()` returns `CLAUDE_TEST_MODEL`, or `haiku` when it isn't set.

## Reference Timings

Measured on 2026-09-30 (Windows 11, CLI 2.1.285):

| Suite | Model | Duration |
|-------|-------|----------|
| Offline (51 tests) | - | ~1 s |
| Live (24 tests) | haiku | ~3 min |
| Single `invoke` | haiku | ~3-4 s (includes ~1-3 s CLI startup) |

## Best Practices

1. **CI**: run offline flows on every change. Run live flows only where a logged-in CLI is available.
2. **Development**: use `haiku` for quick iteration.
3. **Before a release**: run the live suite once with `sonnet`.
