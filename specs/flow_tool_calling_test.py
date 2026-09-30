"""
Live flow test: tool calling against the real Claude Code CLI.
Reference: flow_tool_calling.md
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from claude_code_langchain import ClaudeCodeChatModel

from .test_helpers import get_test_model_name

pytestmark = pytest.mark.live

WEATHER = {"paris": "18C, light rain", "tokyo": "26C, sunny"}
TIME = {"paris": "09:15", "tokyo": "16:15"}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return WEATHER.get(city.lower(), "unknown")


@tool
def get_time(city: str) -> str:
    """Get the current local time for a city."""
    return TIME.get(city.lower(), "unknown")


TOOLS = {"get_weather": get_weather, "get_time": get_time}


@pytest.fixture
def model():
    return ClaudeCodeChatModel(model=get_test_model_name())


def test_model_requests_a_tool_call(model):
    response = model.bind_tools([get_weather, get_time]).invoke("What's the weather in Paris?")

    assert [c["name"] for c in response.tool_calls] == ["get_weather"]
    assert response.tool_calls[0]["args"]["city"].lower() == "paris"
    assert response.tool_calls[0]["id"]
    assert response.response_metadata["stop_reason"] == "tool_use"


def test_model_answers_directly_when_no_tool_is_needed(model):
    response = model.bind_tools([get_weather]).invoke("Say hello in three words.")

    assert response.tool_calls == []
    assert response.content.strip()


def test_manual_tool_loop_reaches_final_answer(model):
    """Run tools, feed results back, and check the model finishes without repeating calls."""
    bound = model.bind_tools([get_weather, get_time])
    messages = [
        SystemMessage(content="You are a concise travel assistant."),
        HumanMessage(content="What's the weather and the local time in Tokyo?"),
    ]
    seen_calls = []

    for _ in range(4):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            key = (call["name"], call["args"]["city"].lower())
            assert key not in seen_calls, f"repeated tool call {key}"
            seen_calls.append(key)
            messages.append(
                ToolMessage(TOOLS[call["name"]].invoke(call["args"]), tool_call_id=call["id"])
            )

    final = messages[-1]
    assert not final.tool_calls, "model never produced a final answer"
    assert "26" in final.content and "16:15" in final.content
    assert {name for name, _ in seen_calls} == {"get_weather", "get_time"}


def test_forced_tool_choice(model):
    response = model.bind_tools([get_weather, get_time], tool_choice="get_time").invoke(
        "Tell me about Paris."
    )

    assert [c["name"] for c in response.tool_calls] == ["get_time"]


def test_with_structured_output(model):
    class Person(BaseModel):
        """Information about a person."""

        name: str = Field(description="Full name")
        age: int
        languages: list[str]

    person = model.with_structured_output(Person).invoke(
        "Maria Lopez is 34 and speaks Spanish, English and French."
    )

    assert person.name == "Maria Lopez"
    assert person.age == 34
    assert set(person.languages) == {"Spanish", "English", "French"}


def test_create_agent_loop(model):
    agents = pytest.importorskip("langchain.agents")
    executed = []

    @tool
    def find_city(landmark: str) -> str:
        """Return the city where a landmark is located."""
        executed.append("find_city")
        return "Zurich"

    @tool
    def city_weather(city: str) -> str:
        """Get the current weather for a city."""
        executed.append(f"city_weather:{city}")
        return "11C, fog"

    agent = agents.create_agent(
        model, tools=[find_city, city_weather], system_prompt="Never guess facts; use tools."
    )
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather at the 'Glass Needle'?"}]},
        {"recursion_limit": 10},
    )

    assert executed == ["find_city", "city_weather:Zurich"]
    assert "fog" in out["messages"][-1].content.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
