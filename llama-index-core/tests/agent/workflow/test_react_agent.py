from typing import List

import pytest
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole, MockLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import ToolSelection
from workflows import Context


def test_react_agent_prompts():
    llm = MockLLM()
    agent = ReActAgent(
        llm=llm,
        tools=[],
    )

    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert isinstance(prompts["react_header"], PromptTemplate)

    new_prompt = "New prompt"
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt in str(prompts["react_header"])

    new_prompt = PromptTemplate("New prompt 2")
    agent.update_prompts({"react_header": new_prompt})
    prompts = agent.get_prompts()
    assert len(prompts) == 1
    assert new_prompt == prompts["react_header"]


@pytest.mark.asyncio
async def test_react_agent_handoff_clears_current_reasoning():
    """A handoff should not leave stale reasoning in the shared context store."""
    llm_inputs: List[List[ChatMessage]] = []
    responses = [
        # turn 1: A (ReAct) hands off to B
        'Thought: not my job\nAction: handoff\nAction Input: {"to_agent": "B", "reason": "B knows"}',
        # turn 1: B answers
        "B has the answer.",
        # turn 2: B hands back to A
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="hand back",
            additional_kwargs={
                "tool_calls": [
                    ToolSelection(
                        tool_id="1",
                        tool_name="handoff",
                        tool_kwargs={"to_agent": "A", "reason": "A should answer"},
                    )
                ]
            },
        ),
        # turn 2: A answers
        "Thought: I can answer now\nAnswer: A says hi",
    ]

    def response_generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        llm_inputs.append(list(messages))
        response = responses.pop(0)
        if isinstance(response, str):
            response = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        return response

    llm = MockFunctionCallingLLM(response_generator=response_generator)
    agent_a = ReActAgent(
        name="A",
        description="Agent A",
        system_prompt="You are agent A.",
        llm=llm,
    )
    agent_b = FunctionAgent(
        name="B",
        description="Agent B",
        system_prompt="You are agent B.",
        llm=llm,
    )

    workflow = AgentWorkflow(agents=[agent_a, agent_b], root_agent="A")
    memory = ChatMemoryBuffer.from_defaults()
    ctx = Context(workflow)

    await workflow.run(user_msg="hello", memory=memory, ctx=ctx)

    # the handoff reasoning should not stay in the shared context store
    current_reasoning = await ctx.store.get("current_reasoning", default=[])
    assert current_reasoning == []

    await workflow.run(user_msg="again", memory=memory, ctx=ctx)

    # the stale turn-1 reasoning should not be re-injected into A's prompt
    assert not any(
        str(message.content).startswith("Observation: Agent B is now handling")
        for message in llm_inputs[-1]
    )

    # A's answer should be recorded without the stale turn-1 reasoning
    live_memory = await ctx.store.get("memory")
    chat_history = await live_memory.aget()
    assert "A says hi" in str(chat_history[-1].content)
    assert "not my job" not in str(chat_history[-1].content)


@pytest.mark.asyncio
async def test_react_agent_receiving_handoff_starts_with_clean_reasoning():
    """A ReAct agent receiving a handoff should not inherit the sender's steps."""
    llm_inputs: List[List[ChatMessage]] = []
    responses = [
        # A (ReAct) hands off to B
        'Thought: not my job\nAction: handoff\nAction Input: {"to_agent": "B", "reason": "B knows"}',
        # B (ReAct) answers
        "Thought: B here\nAnswer: B has the answer.",
    ]

    def response_generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        llm_inputs.append(list(messages))
        return ChatMessage(role=MessageRole.ASSISTANT, content=responses.pop(0))

    llm = MockFunctionCallingLLM(response_generator=response_generator)
    agent_a = ReActAgent(
        name="A",
        description="Agent A",
        system_prompt="You are agent A.",
        llm=llm,
    )
    agent_b = ReActAgent(
        name="B",
        description="Agent B",
        system_prompt="You are agent B.",
        llm=llm,
    )

    workflow = AgentWorkflow(agents=[agent_a, agent_b], root_agent="A")
    memory = ChatMemoryBuffer.from_defaults()

    await workflow.run(user_msg="hello", memory=memory)

    # B's prompt should not contain A's steps injected as B's own reasoning
    assert not any(
        str(message.content).startswith("Observation:") for message in llm_inputs[1]
    )

    # B's recorded answer should not contain A's reasoning steps
    chat_history = await memory.aget()
    assert "B has the answer." in str(chat_history[-1].content)
    assert "not my job" not in str(chat_history[-1].content)
