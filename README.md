# Claude Code SDK - LangChain Adapter

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)](https://github.com/langchain-ai/langchain)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/MrOplus/claude-code-sdk-langchain)

Use Claude through your **Claude Code subscription** as a LangChain chat model, so you can prototype agentic applications **without per-token API charges**.

`ClaudeCodeChatModel` is a regular LangChain `BaseChatModel`. It works with `invoke`, `stream`, `batch`, their async versions, LCEL chains and output parsers. Requests run through the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) and the Claude Code CLI. When you're ready for production, swap in `ChatAnthropic` by changing one line.

## 🎯 Why

- ✅ Prototype LangChain apps on your existing subscription
- ✅ Test agent ideas without worrying about API costs
- ✅ Move to the official API later by changing one line

## 📦 Installation

### Prerequisites

The **Claude Code CLI** must be installed and logged in:

```bash
npm install -g @anthropic-ai/claude-code
claude   # log in once with your subscription
```

### Via GitHub

```bash
# Latest version from main
pip install git+https://github.com/MrOplus/claude-code-sdk-langchain.git

# Specific version tag
pip install git+https://github.com/MrOplus/claude-code-sdk-langchain.git@v0.2.0
```

### Via Pixi

```toml
[pypi-dependencies]
claude-code-langchain = { git = "https://github.com/MrOplus/claude-code-sdk-langchain.git", tag = "v0.2.0" }
```

## 🚀 Quick Start

```python
from claude_code_langchain import ClaudeCodeChatModel
from langchain_core.prompts import ChatPromptTemplate

model = ClaudeCodeChatModel(model="sonnet")   # "haiku", "sonnet", "opus" or a full model ID

response = model.invoke("What is LangChain?")
print(response.content)
print(response.usage_metadata)                 # {'input_tokens': ..., 'output_tokens': ..., ...}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}"),
    ("human", "{question}"),
])
chain = prompt | model
result = chain.invoke({"domain": "Python", "question": "How do I create a REST API?"})
```

## 🔄 Streaming

Streaming is **token-by-token**, not block-by-block:

```python
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

async for chunk in model.astream("List 5 ideas"):
    print(chunk.content, end="", flush=True)
```

If you stop iterating early, the CLI request is interrupted, so an abandoned stream doesn't keep generating and using your quota.

## 🔒 Isolation by Default

A plain `ClaudeCodeChatModel()` behaves like a chat API, not like a coding agent:

- **No tools**: Claude Code's built-in tools are disabled, and so are MCP servers, including
  claude.ai account connectors such as Gmail or Drive. Without this, connector tool
  definitions add thousands of input tokens to every request and change the model's answers.
- **No local settings**: `CLAUDE.md` files, hooks and permission settings aren't loaded.
- **No transcripts**: sessions aren't saved under `~/.claude`.

Each of these can be turned back on with `builtin_tools`, `mcp_servers` / `strict_mcp_config`,
`setting_sources` and `persist_session`.

## ⚙️ Configuration

```python
model = ClaudeCodeChatModel(
    model="sonnet",                    # alias or full model ID (default: "sonnet")
    system_prompt="You are an expert", # default system prompt (a SystemMessage overrides it)
    stop=["\n\n"],                     # default stop sequences (emulated client-side)
    effort="medium",                   # reasoning effort: low | medium | high | xhigh | max
    timeout=120,                       # seconds per request (raises ClaudeCodeTimeoutError)

    # Claude Code specifics - defaults give a plain, isolated chat model
    builtin_tools=[],                  # built-in tools to enable, e.g. ["WebSearch"]
    allowed_tools=[],                  # tools that run without a permission prompt
    max_turns=1,                       # raise this when enabling tools
    setting_sources=[],                # [] = ignore CLAUDE.md/hooks/settings; None = load all
    mcp_servers={},                    # MCP servers to attach explicitly
    strict_mcp_config=True,            # ignore MCP servers from settings and claude.ai connectors
    persist_session=False,             # don't write session transcripts to ~/.claude
    permission_mode=None,              # default | acceptEdits | plan | bypassPermissions
    cwd=None,                          # working directory for the CLI
    env={},                            # extra environment variables for the CLI
    cli_path=None,                     # explicit path to the `claude` executable

    # Accepted for ChatAnthropic compatibility, but NOT supported (a warning is logged)
    temperature=None,
    max_tokens=None,
)
```

### Response metadata

| Field | Content |
|-------|---------|
| `response.usage_metadata` | `input_tokens` (including cache reads/writes, like ChatAnthropic), `output_tokens`, `total_tokens`, cache and reasoning details |
| `response.response_metadata` | `model_name` (resolved model ID), `stop_reason`, `session_id`, `cost_usd`, `duration_ms`, raw `usage` |
| `response.additional_kwargs["thinking"]` | The model's thinking text, when present |

## 🔄 Moving to Production

```python
# Development (your subscription)
from claude_code_langchain import ClaudeCodeChatModel
model = ClaudeCodeChatModel(model="sonnet")

# Production (official API)
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-5-5", api_key="sk-...")
```

The rest of your code stays the same.

## 🏗️ Architecture

```
LangChain app
     ↓
ClaudeCodeChatModel      (this adapter: messages → prompt, SDK events → LangChain chunks)
     ↓
claude-agent-sdk         (ClaudeSDKClient, one short-lived session per request)
     ↓
Claude Code CLI          (authenticated with your subscription)
     ↓
Claude
```

Each request starts a CLI process. The model is **stateless**, like every LangChain chat model: conversation history is whatever messages you pass in.

## ⚠️ Limitations

These are deliberate trade-offs for free prototyping. Where behavior differs from `ChatAnthropic`, the adapter logs a warning.

| Feature | Behavior | Why / workaround |
|---------|----------|------------------|
| `temperature` | Ignored, with a warning | The CLI has no sampling controls. Use `ChatAnthropic` if you need them. |
| `max_tokens` | Ignored, with a warning | The CLI's output limit fails the request instead of truncating it, so it can't be emulated faithfully. |
| `stop` sequences | Emulated | Output is truncated at the first match and generation is interrupted. `stop_reason` is `"stop_sequence"`. |
| Images / files | Dropped, with a warning | The CLI prompt channel is text-only. Use `ChatAnthropic` for vision. |
| Native tool calling (`bind_tools`) | Not supported | Use prompting, or enable Claude Code's own `builtin_tools`. |
| Multi-turn history | Rendered as a transcript | The CLI takes one user turn per request, so earlier turns are sent as `Human:` / `Assistant:` text. A single human message is sent verbatim. |
| Latency | Higher than the API | Each request starts a CLI process (about 1–3 s of overhead). |
| Environment context | ~400 input tokens | The CLI always adds basic environment info (OS, shell, working directory, date) to the context. It can't be turned off when a custom system prompt is used. |
| Quotas | Your subscription limits | Switch to the API if you hit them. |

### System prompt precedence

`SystemMessage`s in the input become the real system prompt. If you also set `system_prompt` in the constructor, the message wins and a warning is logged.

### Stopping a stream early

LangChain's `stream()` / `astream()` wrappers don't close the model's generator when you `break`. The adapter's cleanup (interrupting the CLI) runs when that generator is garbage-collected, which in CPython usually happens right away. In a long-running event loop this happens within about a second. If your script exits immediately after abandoning a stream, the SDK's exit handler terminates the CLI process.

## 🧪 Tests

```bash
pixi run test-offline   # fast, deterministic, no CLI or subscription needed
pixi run test-live      # real CLI calls (uses CLAUDE_TEST_MODEL, default "haiku")
pixi run test           # everything
pixi run smoke          # one-shot end-to-end check
```

Without pixi: `pip install -e ".[dev]"`, then run `pytest specs -m "not live"`. Live tests are skipped automatically when the `claude` CLI isn't on your PATH, or when `CLAUDE_SKIP_LIVE=1` is set. See [`specs/README.md`](specs/README.md).

## 📝 Examples

[`examples/basic_usage.py`](examples/basic_usage.py) covers invocation, system prompts, streaming, chains, async, batch, routing, multi-turn history and stop sequences.

## 🤝 Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT

## 🙏 Acknowledgments

- Anthropic for Claude, Claude Code and the Claude Agent SDK
- LangChain for the framework
- Stéphane Wootha Richard for the original adapter
