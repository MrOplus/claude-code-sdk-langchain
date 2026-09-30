# __INDEX - src

## claude_code_langchain/
Description: Main package - LangChain chat model backed by claude-agent-sdk
Created: 2025-09-29
Modified: 2026-09-30

### claude_code_langchain/__init__.py
Description: Package exports - ClaudeCodeChatModel, ClaudeCodeError, ClaudeCodeTimeoutError, DEFAULT_MODEL, MessageConverter
Created: 2025-09-29
Modified: 2026-09-30

### claude_code_langchain/chat_model.py
Description: ClaudeCodeChatModel - ClaudeSDKClient lifecycle (interrupt/disconnect on early exit), token-level streaming, tool calling via deferred in-process MCP tools, stop sequence emulation, usage metadata, anyio isolation, timeouts
Created: 2025-09-29
Modified: 2026-09-30

### claude_code_langchain/message_converter.py
Description: LangChain messages to system prompt + user prompt (structured conversation history incl. tool calls/results), multimodal text extraction, usage/response metadata conversion
Created: 2025-09-29
Modified: 2026-09-30
