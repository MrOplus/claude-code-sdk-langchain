"""
Offline flow test: tool calling (bind_tools, tool_choice, with_structured_output).
Reference: flow_tool_calling.md
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from claude_code_langchain import ClaudeCodeChatModel

from .fake_sdk import assistant, result, streamed_reply, text_delta, tool_use


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "18C"


@tool
def get_time(city: str) -> str:
    """Get the current local time for a city."""
    return "09:15"


class Person(BaseModel):
    """Information about a person."""

    name: str
    age: int


def test_tool_call_is_returned_not_executed(fake_claude):
    fake_claude.script = [tool_use("get_weather", {"city": "Paris"}, "toolu_1"), result()]
    model = ClaudeCodeChatModel().bind_tools([get_weather, get_time])

    response = model.invoke("Weather in Paris?")

    assert response.tool_calls == [
        {"name": "get_weather", "args": {"city": "Paris"}, "id": "toolu_1", "type": "tool_call"}
    ]
    assert response.response_metadata["stop_reason"] == "tool_use"


def test_bound_tools_are_exposed_as_deferred_native_tools(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().bind_tools([get_weather, get_time]).invoke("Hi")

    options = fake_claude.instances[0].options
    assert "langchain" in options.mcp_servers
    assert "mcp__langchain__get_weather" in options.allowed_tools
    assert "mcp__langchain__get_time" in options.allowed_tools
    matcher = options.hooks["PreToolUse"][0]
    decision = asyncio.run(matcher.hooks[0]({}, "toolu_1", None))
    assert decision["hookSpecificOutput"]["permissionDecision"] == "defer"


def test_no_tool_setup_without_bound_tools(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().invoke("Hi")

    options = fake_claude.instances[0].options
    assert options.mcp_servers == {}
    assert options.hooks is None


def test_parallel_tool_calls(fake_claude):
    fake_claude.script = [
        tool_use("get_weather", {"city": "Tokyo"}, "toolu_1"),
        tool_use("get_time", {"city": "Tokyo"}, "toolu_2"),
        result(),
    ]
    response = ClaudeCodeChatModel().bind_tools([get_weather, get_time]).invoke("Tokyo?")

    assert [(c["name"], c["id"]) for c in response.tool_calls] == [
        ("get_weather", "toolu_1"),
        ("get_time", "toolu_2"),
    ]


def test_text_and_tool_call_together(fake_claude):
    fake_claude.script = [
        text_delta("Let me check."),
        assistant(text="Let me check."),
        tool_use("get_weather", {"city": "Paris"}, "toolu_1"),
        result(),
    ]
    response = ClaudeCodeChatModel().bind_tools([get_weather]).invoke("Paris?")

    assert response.content == "Let me check."
    assert response.tool_calls[0]["name"] == "get_weather"


def test_builtin_tool_use_is_not_reported_as_tool_call(fake_claude):
    """Claude Code's own tools (e.g. WebSearch) run inside the CLI, not in LangChain."""
    fake_claude.script = [
        tool_use("WebSearch", {"query": "x"}, "toolu_9", server=""),
        *streamed_reply("done"),
    ]
    response = ClaudeCodeChatModel().bind_tools([get_weather]).invoke("Search")

    assert response.tool_calls == []
    assert response.content == "done"


def test_streaming_tool_call_chunks_aggregate(fake_claude):
    fake_claude.script = [tool_use("get_time", {"city": "Paris"}, "toolu_1"), result()]
    chunks = list(ClaudeCodeChatModel().bind_tools([get_time]).stream("Time?"))

    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    assert total.tool_calls[0]["args"] == {"city": "Paris"}
    assert total.tool_calls[0]["id"] == "toolu_1"


def test_tool_choice_any_adds_instruction(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().bind_tools([get_weather], tool_choice="any").invoke(
        [SystemMessage(content="Be terse."), HumanMessage(content="Hi")]
    )

    system_prompt = fake_claude.instances[0].options.system_prompt
    assert system_prompt.startswith("Be terse.")
    assert "must respond by calling" in system_prompt


def test_forced_tool_is_the_only_tool_exposed(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().bind_tools([get_weather, get_time], tool_choice="get_time").invoke("Hi")

    options = fake_claude.instances[0].options
    tool_names = [t for t in options.allowed_tools if t.startswith("mcp__langchain__")]
    assert tool_names == ["mcp__langchain__get_time"]
    assert "`get_time`" in options.system_prompt


def test_openai_style_forced_tool_choice(fake_claude):
    fake_claude.script = streamed_reply("ok")
    choice = {"type": "function", "function": {"name": "get_weather"}}
    ClaudeCodeChatModel().bind_tools([get_weather, get_time], tool_choice=choice).invoke("Hi")

    tools = [t for t in fake_claude.instances[0].options.allowed_tools]
    assert tools == ["mcp__langchain__get_weather"]


def test_tool_choice_none_disables_tools(fake_claude):
    fake_claude.script = streamed_reply("ok")
    ClaudeCodeChatModel().bind_tools([get_weather], tool_choice="none").invoke("Hi")

    assert fake_claude.instances[0].options.mcp_servers == {}


def test_with_structured_output_returns_pydantic_object(fake_claude):
    fake_claude.script = [tool_use("Person", {"name": "Maria", "age": 34}, "toolu_1"), result()]
    structured = ClaudeCodeChatModel().with_structured_output(Person)

    person = structured.invoke("Maria is 34.")

    assert person == Person(name="Maria", age=34)
    options = fake_claude.instances[0].options
    assert options.allowed_tools == ["mcp__langchain__Person"]


def test_with_structured_output_include_raw(fake_claude):
    fake_claude.script = [tool_use("Person", {"name": "Li", "age": 5}, "toolu_1"), result()]
    out = ClaudeCodeChatModel().with_structured_output(Person, include_raw=True).invoke("Li is 5.")

    assert out["parsed"] == Person(name="Li", age=5)
    assert isinstance(out["raw"], AIMessage)
    assert out["parsing_error"] is None


def test_tool_results_are_replayed_as_structured_history(fake_claude):
    fake_claude.script = streamed_reply("It is 18C in Paris.")
    history = [
        HumanMessage(content="Weather in Paris?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "toolu_1"}],
        ),
        ToolMessage(content="18C", tool_call_id="toolu_1", name="get_weather"),
    ]

    response = ClaudeCodeChatModel().bind_tools([get_weather]).invoke(history)

    prompt = fake_claude.instances[0].prompts[0]
    assert '<tool_call id="toolu_1" name="get_weather">{"city": "Paris"}</tool_call>' in prompt
    assert '<tool_result tool_call_id="toolu_1" name="get_weather">\n18C\n</tool_result>' in prompt
    assert "Do not repeat those calls" in prompt
    assert response.content == "It is 18C in Paris."
    assert response.tool_calls == []
