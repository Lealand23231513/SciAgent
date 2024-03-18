# 核心控制模块
from typing import cast
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.adapters.openai import convert_dict_to_message
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import APIStatusError
from global_var import get_global_value, set_global_value
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_zhipu import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import load_qwen_agent_executor
from tools import ToolsState
from llm_state import LLMState
from channel import load_channel

logger = logging.getLogger(Path(__file__).stem)


def load_openai_agent_excutor(
    tools_inst: list[BaseTool], model="gpt-3.5-turbo", api_key=None, base_url=None
):
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key, base_url=base_url)
    if len(tools_inst) == 0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst, tool_choice="auto")
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_inst, handle_parsing_errors=True  # type: ignore
    )
    return agent_executor


def load_zhipuai_agent_excutor(
    tools_inst: list[BaseTool], model="glm-3-turbo", api_key=None, base_url=None
):
    if model == "chatglm3-6b":
        api_key = "EMP.TY"
    llm = ChatZhipuAI(model=model, temperature=0.01, api_key=api_key, base_url=base_url)
    if len(tools_inst) == 0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst, tool_choice="auto")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant.",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_inst, handle_parsing_errors=True  # type: ignore
    )
    return agent_executor


def call_agent(user_input: str, stream: bool = False):
    load_dotenv()
    llm_state = cast(LLMState, get_global_value("llm_state"))
    tools_state = cast(ToolsState, get_global_value("tools_state"))
    agent_excutor_mapping = {
        "openai": load_openai_agent_excutor,
        "zhipuai": load_zhipuai_agent_excutor,
        "qwen": load_qwen_agent_executor,
    }
    model_kwargs = llm_state.model_dump()
    if "gpt" in llm_state.model:
        agent_executor = agent_excutor_mapping["openai"](
            tools_state.tools_inst, **model_kwargs
        )
    elif "glm" in llm_state.model:
        agent_executor = agent_excutor_mapping["zhipuai"](
            tools_state.tools_inst, **model_kwargs
        )
    elif "qwen" in llm_state.model:
        agent_executor = agent_excutor_mapping["qwen"](
            tools_state.tools_inst, **model_kwargs
        )
    set_global_value("agent_executor", agent_executor)
    chat_history = cast(list[dict[str, str]], get_global_value("chat_history"))
    try:
        ans = agent_executor.invoke(
            {
                "chat_history": [convert_dict_to_message(m) for m in chat_history],
                "input": user_input,
            }
        )
        logger.info({k: ans[k] for k in ("input", "output")})
        if stream:
            # fake stream
            yield from ans["output"]
        else:
            return ans["output"]
    except APIStatusError as e:
        channel = load_channel()
        msg = json.dumps(
            {
                "type": "modal",
                "name": "error",
                "message": repr(e),
            }
        )
        channel.send(msg, this='back')
        if stream:
            yield from f"An error raised: {repr(e)}"
        else:
            return f"An error raised: {repr(e)}"
