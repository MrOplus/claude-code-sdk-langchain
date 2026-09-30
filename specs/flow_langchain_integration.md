# Flow: Integration with LangChain Chains

## Description
Tests using ClaudeCodeChatModel in LangChain chains (LCEL) against the real Claude Code CLI.

## Flow
1. Create a ClaudeCodeChatModel
2. Create a ChatPromptTemplate with system and human messages
3. Build a chain with the pipe operator (|)
4. Invoke the chain with variables
5. Verify that the chain works correctly
6. Add an output parser

## Test Cases
- Simple chain: `prompt | model`
- Chain with parser: `prompt | model | StrOutputParser()`
- Async chain: `await chain.ainvoke(...)`
- Batch processing: `model.batch([...])` with independent, verifiable answers
- Multi-step chain: classify sentiment, then answer with `RunnablePassthrough.assign`

## Validation
- Chains run without errors
- Variables are substituted correctly
- Parsers return strings
- Batch answers match their own questions
