# Flow: Basic Chat with ClaudeCodeChatModel

## Description
Tests the basic chat journey against the real Claude Code CLI.

## Flow
1. Create a ClaudeCodeChatModel instance
2. Send a simple message ("Hello, who are you?")
3. Receive and validate the response
4. Check that the response is an AIMessage
5. Confirm that the content isn't empty

## Additional Journeys
- Async invocation with `ainvoke()`
- Response metadata: `usage_metadata` token counts, the resolved `model_name`, `session_id`
- Isolation: default requests carry no tool or connector definitions (input tokens stay small)
- System messages are followed
- Conversation history is used (multi-turn)
- Stop sequences truncate the output and report `stop_reason="stop_sequence"`

## Validation
- The model responds correctly
- The response is a valid AIMessage with usage metadata
- The content contains text
- Nothing errors during execution
