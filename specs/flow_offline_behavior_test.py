"""
Offline flow tests: adapter behavior through the public API with a scripted SDK client.
Reference: flow_offline_behavior.md
"""

import gc
import logging

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from claude_code_langchain import ClaudeCodeChatModel, ClaudeCodeTimeoutError

from .fake_sdk import assistant, result, streamed_reply, text_delta


def test_invoke_returns_message_with_usage_and_metadata(fake_claude):
    fake_claude.script = streamed_reply("Hello", " there", thinking="Let me think")
    model = ClaudeCodeChatModel(model="haiku")

    response = model.invoke("Hi")

    assert isinstance(response, AIMessage)
    assert response.content == "Hello there"
    assert response.additional_kwargs["thinking"] == "Let me think"
    assert response.additional_kwargs["model"] == "claude-test-1"

    meta = response.response_metadata
    assert meta["model_name"] == "claude-test-1"
    assert meta["session_id"] == "session-123"
    assert meta["stop_reason"] == "end_turn"
    assert meta["cost_usd"] == pytest.approx(0.001)
    assert meta["duration_ms"] == 120

    # Mirrors ChatAnthropic: input_tokens includes cache reads/writes
    assert response.usage_metadata["input_tokens"] == 15
    assert response.usage_metadata["output_tokens"] == 7
    assert response.usage_metadata["total_tokens"] == 22
    assert response.usage_metadata["input_token_details"]["cache_read"] == 5
    assert response.usage_metadata["output_token_details"]["reasoning"] == 3


def test_default_options_make_a_pure_isolated_chat_model(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().invoke("Hi")

    options = fake_claude.instances[0].options
    assert options.model == "sonnet"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {}
    assert options.max_turns == 1
    assert options.include_partial_messages is True
    assert "no-session-persistence" in options.extra_args
    assert options.system_prompt is None


def test_configuration_is_forwarded_to_the_sdk(fake_claude):
    fake_claude.script = streamed_reply("ok")
    model = ClaudeCodeChatModel(
        model="opus",
        effort="high",
        builtin_tools=["WebSearch"],
        allowed_tools=["WebSearch"],
        max_turns=3,
        setting_sources=None,
        mcp_servers={"docs": {"type": "http", "url": "https://example.com/mcp"}},
        strict_mcp_config=False,
        persist_session=True,
        env={"FOO": "bar"},
        cwd="/tmp",
    )
    model.invoke("Hi")

    options = fake_claude.instances[0].options
    assert options.model == "opus"
    assert options.effort == "high"
    assert options.tools == ["WebSearch"]
    assert options.allowed_tools == ["WebSearch"]
    assert options.max_turns == 3
    assert options.setting_sources is None
    assert options.mcp_servers == {"docs": {"type": "http", "url": "https://example.com/mcp"}}
    assert options.strict_mcp_config is False
    assert "no-session-persistence" not in options.extra_args
    assert options.env == {"FOO": "bar"}
    assert options.cwd == "/tmp"


def test_single_human_message_is_sent_verbatim(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().invoke([HumanMessage(content="What is 2+2?")])

    assert fake_claude.instances[0].prompts == ["What is 2+2?"]


def test_system_messages_become_the_system_prompt(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().invoke(
        [SystemMessage(content="Be terse."), HumanMessage(content="Hello")]
    )

    client = fake_claude.instances[0]
    assert client.options.system_prompt == "Be terse."
    assert client.prompts == ["Hello"]


def test_system_message_takes_precedence_over_constructor_prompt(fake_claude, caplog):
    fake_claude.script = streamed_reply("ok")
    model = ClaudeCodeChatModel(system_prompt="Constructor prompt")

    with caplog.at_level(logging.WARNING):
        model.invoke([SystemMessage(content="Message prompt"), HumanMessage(content="Hi")])

    assert fake_claude.instances[0].options.system_prompt == "Message prompt"
    assert "takes precedence" in caplog.text


def test_constructor_system_prompt_is_used_without_system_message(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel(system_prompt="Constructor prompt").invoke("Hi")

    assert fake_claude.instances[0].options.system_prompt == "Constructor prompt"


def test_multi_turn_history_is_rendered_as_transcript(fake_claude):
    fake_claude.script = streamed_reply("Your name is Alice.")
    ClaudeCodeChatModel().invoke(
        [
            HumanMessage(content="My name is Alice"),
            AIMessage(content="Nice to meet you, Alice!"),
            HumanMessage(content="What is my name?"),
        ]
    )

    prompt = fake_claude.instances[0].prompts[0]
    assert prompt.startswith("<conversation_history>\n<user>\nMy name is Alice\n</user>")
    assert "<assistant>\nNice to meet you, Alice!\n</assistant>" in prompt
    assert prompt.endswith("</conversation_history>\n\nReply to the latest user message.")


def test_images_are_dropped_with_warning(fake_claude, caplog):
    fake_claude.script = streamed_reply("ok")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]
    )

    with caplog.at_level(logging.WARNING):
        ClaudeCodeChatModel().invoke([message])

    assert fake_claude.instances[0].prompts == ["Describe this image"]
    assert "NOT SUPPORTED" in caplog.text


def test_unsupported_sampling_parameters_warn_once(fake_claude, caplog):
    with caplog.at_level(logging.WARNING):
        ClaudeCodeChatModel(temperature=0.2, max_tokens=100)
    assert "temperature=0.2" in caplog.text
    assert "max_tokens=100" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ClaudeCodeChatModel()
    assert caplog.text == ""


def test_sync_stream_yields_token_level_chunks(fake_claude):
    fake_claude.script = streamed_reply("One", ", two", ", three")
    chunks = list(ClaudeCodeChatModel().stream("Count"))

    texts = [c.content for c in chunks if c.content]
    assert texts == ["One", ", two", ", three"]

    # Chunks aggregate into a complete message with usage metadata
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    assert total.content == "One, two, three"
    assert total.usage_metadata["output_tokens"] == 7


async def test_async_stream_yields_token_level_chunks(fake_claude):
    fake_claude.script = streamed_reply("A", "B", "C")
    texts = [c.content async for c in ClaudeCodeChatModel().astream("Letters") if c.content]

    assert texts == ["A", "B", "C"]


def test_full_blocks_are_used_when_no_partial_events_arrive(fake_claude):
    fake_claude.script = [assistant(text="Complete answer"), result()]

    assert ClaudeCodeChatModel().invoke("Hi").content == "Complete answer"
    fake_claude.reset([assistant(text="Complete answer"), result()])
    assert "".join(c.content for c in ClaudeCodeChatModel().stream("Hi")) == "Complete answer"


def test_subagent_output_is_not_mixed_into_the_reply(fake_claude):
    fake_claude.script = [text_delta("subagent noise", parent="tool-1"), *streamed_reply("Main")]

    assert ClaudeCodeChatModel().invoke("Hi").content == "Main"


def test_stop_sequence_truncates_invoke_and_interrupts_cli(fake_claude):
    fake_claude.script = streamed_reply("1 2 3 ", "4 5 6", " 7 8")
    response = ClaudeCodeChatModel().invoke("Count", stop=["5"])

    assert response.content == "1 2 3 4 "
    assert response.response_metadata["stop_reason"] == "stop_sequence"
    client = fake_claude.instances[0]
    assert client.interrupted and client.disconnected


def test_stop_sequence_split_across_chunks_never_leaks(fake_claude):
    fake_claude.script = streamed_reply("Hello EN", "D and more")
    chunks = [c.content for c in ClaudeCodeChatModel(stop=["END"]).stream("Hi")]

    assert "".join(chunks) == "Hello "
    assert not any("EN" in c for c in chunks)


def test_early_break_interrupts_and_disconnects(fake_claude):
    fake_claude.script = streamed_reply(*[f"w{i} " for i in range(50)])
    fake_claude.delay = 0.03  # token pacing above Windows timer resolution (~15.6ms)
    for i, _ in enumerate(ClaudeCodeChatModel().stream("Long story")):
        if i == 2:
            break
    # LangChain's stream() wrapper does not close the model's generator on break;
    # it is finalized when collected (callback tracers can hold it in a cycle).
    gc.collect()

    client = fake_claude.instances[0]
    assert client.interrupted, "abandoned request should be interrupted"
    assert client.disconnected
    assert client.delivered < 50


def test_completed_request_disconnects_without_interrupt(fake_claude):
    fake_claude.script = streamed_reply("done")
    ClaudeCodeChatModel().invoke("Hi")

    client = fake_claude.instances[0]
    assert client.disconnected and not client.interrupted


def test_streaming_callbacks_receive_tokens(fake_claude):
    fake_claude.script = streamed_reply("a", "b", "c")

    class Collector(BaseCallbackHandler):
        def __init__(self):
            self.tokens = []

        def on_llm_new_token(self, token, **kwargs):
            if token:  # LangChain also dispatches empty metadata chunks
                self.tokens.append(token)

    collector = Collector()
    list(ClaudeCodeChatModel().stream("Hi", config={"callbacks": [collector]}))

    assert collector.tokens == ["a", "b", "c"]


async def test_async_chain_with_parser_streams_strings(fake_claude):
    fake_claude.script = streamed_reply("Par", "is")
    prompt = ChatPromptTemplate.from_messages([("system", "Be terse"), ("human", "{q}")])
    chain = prompt | ClaudeCodeChatModel() | StrOutputParser()

    parts = [p async for p in chain.astream({"q": "Capital of France?"})]

    assert "".join(parts) == "Paris"
    assert all(isinstance(p, str) for p in parts)
    assert fake_claude.instances[0].options.system_prompt == "Be terse"


async def test_sync_invoke_works_inside_running_event_loop(fake_claude):
    fake_claude.script = streamed_reply("from a thread")
    # e.g. Jupyter: a loop is already running when sync invoke() is called
    assert ClaudeCodeChatModel().invoke("Hi").content == "from a thread"


def test_timeout_raises_dedicated_error_and_cleans_up(fake_claude):
    fake_claude.script = streamed_reply("slow", "reply")
    fake_claude.delay = 1.0
    model = ClaudeCodeChatModel(timeout=0.1)

    with pytest.raises(ClaudeCodeTimeoutError) as exc_info:
        model.invoke("Hi")

    assert isinstance(exc_info.value, TimeoutError)
    assert fake_claude.instances[0].disconnected


def test_batch_runs_independent_requests(fake_claude):
    fake_claude.script = streamed_reply("same")
    responses = ClaudeCodeChatModel().batch(["a", "b", "c"])

    assert [r.content for r in responses] == ["same"] * 3
    assert len(fake_claude.instances) == 3


def test_langsmith_params_identify_provider_and_model():
    model = ClaudeCodeChatModel(model="haiku")
    params = model._get_ls_params(stop=["x"])

    assert params["ls_provider"] == "anthropic"
    assert params["ls_model_name"] == "haiku"
    assert params["ls_stop"] == ["x"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
