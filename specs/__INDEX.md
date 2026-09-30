# __INDEX - specs

## README.md
Description: Pragmatic Flow Testing philosophy, offline vs. live flows, how to run and add tests
Created: 2025-09-30
Modified: 2026-09-30

## TEST_CONFIGURATION.md
Description: Live test model selection (CLAUDE_TEST_MODEL), skipping live tests, reference timings
Created: 2025-10-01
Modified: 2026-09-30

## conftest.py
Description: pytest fixtures - `fake_claude` scripted SDK client, automatic skipping of `live` tests
Created: 2026-09-30
Modified: 2026-09-30

## fake_sdk.py
Description: Scripted ClaudeSDKClient stand-in and builders for real SDK message types, including tool use (offline flows)
Created: 2026-09-30
Modified: 2026-09-30

## test_helpers.py
Description: get_test_model_name() - live test model from CLAUDE_TEST_MODEL (default haiku)
Created: 2025-10-01
Modified: 2026-09-30

## flow_basic_chat.md + flow_basic_chat_test.py
Description: [live] Basic chat - sync/async invoke, metadata, system messages, history, stop sequences
Created: 2025-09-30
Modified: 2026-09-30

## flow_langchain_integration.md + flow_langchain_integration_test.py
Description: [live] LangChain integration - LCEL chains, output parsers, async, batch, multi-step chains
Created: 2025-09-30
Modified: 2026-09-30

## flow_streaming.md + flow_streaming_test.py
Description: [live] Streaming - token-level sync/async streaming, aggregation, chain streaming, cancellation
Created: 2025-09-30
Modified: 2026-09-30

## flow_error_handling.md + flow_error_handling_test.py
Description: [offline] Error handling - missing SDK/CLI, process/JSON/connection errors, error results, recovery, invalid input
Created: 2025-09-30
Modified: 2026-09-30

## flow_tool_calling.md + flow_tool_calling_offline_test.py + flow_tool_calling_test.py
Description: [offline + live] Tool calling - bind_tools, parallel calls, tool loop, tool_choice, with_structured_output, create_agent
Created: 2026-09-30
Modified: 2026-09-30

## flow_offline_behavior.md + flow_offline_behavior_test.py
Description: [offline] Adapter behavior - conversion, options, metadata, streaming, stop sequences, early-exit cleanup, timeouts
Created: 2026-09-30
Modified: 2026-09-30
