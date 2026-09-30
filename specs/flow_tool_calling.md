# Flow: Tool Calling

## Description
Validates LangChain tool calling (`bind_tools`, `tool_choice`, `with_structured_output`)
through the Claude Code CLI. Bound tools are exposed to Claude as native tools on an
in-process MCP server. A PreToolUse hook **defers** every call, so the CLI stops and returns
the call instead of executing it. Tool execution stays with the LangChain application, the
same as with ChatAnthropic.

- Offline flows: `flow_tool_calling_offline_test.py` (scripted SDK client)
- Live flows: `flow_tool_calling_test.py` (real CLI)

## User Journeys

### Requesting a Tool Call
1. User binds tools with `model.bind_tools([...])`
2. User asks something that needs a tool
3. User receives an `AIMessage` with `tool_calls` (name, args, id) and
   `stop_reason="tool_use"`; the tool did not run

### Answering Without Tools
1. User asks something that needs no tool
2. User receives a normal text answer with no tool calls

### Parallel Calls
1. User asks for several pieces of information at once
2. User receives several tool calls in one message

### Completing the Loop
1. User runs the requested tools and appends `ToolMessage`s with the results
2. User invokes the model again with the full history
3. The model answers from the results **without repeating the same calls**. It may call
   other tools when it still needs information (sequential dependencies)

### Forcing a Tool
1. User binds with `tool_choice="any"`: the model must call some tool (by instruction)
2. User binds with `tool_choice="<name>"`: only that tool is exposed, so it's the only
   possible call
3. User binds with `tool_choice="none"`: no tools are exposed

### Structured Output
1. User calls `model.with_structured_output(PydanticModel)`
2. User receives a validated Pydantic instance (or a dict for JSON-schema input)

### Agents
1. User passes the model to `langchain.agents.create_agent` (LangGraph)
2. The agent runs tool rounds until the model answers, then stops

### Streaming
1. User streams a response that contains tool calls
2. Chunks aggregate into a message with complete `tool_calls`

## Success Criteria
- Tool calls use Claude's native tool use (no text parsing)
- Bound tools never execute inside the CLI
- Claude Code built-in tools (if enabled) are not reported as LangChain tool calls
- Replayed tool results never cause the same calls to be repeated
