"""
Live flow test: ClaudeCodeChatModel inside LangChain chains (LCEL).
Reference: flow_langchain_integration.md
"""

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from claude_code_langchain import ClaudeCodeChatModel

from .test_helpers import get_test_model_name

pytestmark = pytest.mark.live


@pytest.fixture
def model():
    return ClaudeCodeChatModel(model=get_test_model_name())


def test_simple_chain(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You answer questions about {language} in one sentence."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model

    response = chain.invoke({"language": "Python", "question": "What is a list comprehension?"})

    assert response.content.strip()


def test_chain_with_parser(model):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Always respond very concisely."), ("human", "{input}")]
    )
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({"input": "Define Python in 5 words maximum."})

    assert isinstance(response, str)
    assert response.strip()


async def test_async_chain(model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert in {domain}. Answer in two sentences."),
            ("human", "Explain {concept}"),
        ]
    )
    chain = prompt | model | StrOutputParser()

    response = await chain.ainvoke({"domain": "computer science", "concept": "algorithms"})

    assert isinstance(response, str)
    assert response.strip()


def test_batch_processing(model):
    responses = model.batch(
        [
            [HumanMessage(content="What is 2+2? Reply with just the number.")],
            [HumanMessage(content="Capital of France? One word.")],
            [HumanMessage(content="Color of a clear daytime sky? One word.")],
        ]
    )

    assert len(responses) == 3
    assert "4" in responses[0].content
    assert "paris" in responses[1].content.lower()
    assert "blue" in responses[2].content.lower()


def test_multi_step_chain(model):
    classify = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify the text as exactly one word: positive, negative, or neutral.",
                ),
                ("human", "{text}"),
            ]
        )
        | model
        | StrOutputParser()
    )
    respond = ChatPromptTemplate.from_messages(
        [
            ("system", "The sentiment is {sentiment}. Reply with one short sentence."),
            ("human", "How should I react to: {text}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(sentiment=lambda x: classify.invoke({"text": x["text"]}))
        | respond
        | model
        | StrOutputParser()
    )

    response = chain.invoke({"text": "This product is absolutely fantastic!"})

    assert isinstance(response, str)
    assert response.strip()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
