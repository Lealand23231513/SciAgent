# 核心控制模块
from typing import cast, Any
import os
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.adapters.openai import convert_dict_to_message
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError, OpenAIError, APIConnectionError
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
from model_state import LLMState
from channel import load_channel
AGENT_START = '[AGENT START]'
AGENT_DONE = '[AGENT DONE]'
SEP_OF_LINE = "".join(["-" for _ in range(8)])
logger = logging.getLogger(Path(__file__).stem)


def load_openai_agent_excutor(
    tools_inst: list[BaseTool], model_kwargs:dict[str,Any]
):
    model = model_kwargs.pop('model')
    temperature = model_kwargs.pop('temperature')
    api_key = model_kwargs.pop('api_key')
    base_url = model_kwargs.pop('base_url')
    llm = ChatOpenAI(model=model, temperature=temperature,api_key=api_key,base_url=base_url, model_kwargs=model_kwargs)
    if len(tools_inst) == 0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst)
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
    tools_inst: list[BaseTool], model_kwargs:dict[str,Any]
):
    model = model_kwargs.pop('model')
    temperature = model_kwargs.pop('temperature')
    api_key = model_kwargs.pop('api_key')
    base_url = model_kwargs.pop('base_url')
    llm = ChatOpenAI(model=model, temperature=temperature,api_key=api_key,base_url=base_url, model_kwargs=model_kwargs)
    if model == "chatglm3-6b":
        api_key = "EMP.TY"
    llm = ChatZhipuAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)
    if len(tools_inst) == 0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst)
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


def call_agent(user_input: str):
    # load_dotenv()
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
            tools_state.tools_inst, model_kwargs
        )
    elif "glm" in llm_state.model:
        agent_executor = agent_excutor_mapping["zhipuai"](
            tools_state.tools_inst, model_kwargs
        )
    elif "qwen" in llm_state.model:
        agent_executor = agent_excutor_mapping["qwen"](
            tools_state.tools_inst, model_kwargs
        )
    set_global_value("agent_executor", agent_executor)
    chat_history = cast(list[dict[str, str]], get_global_value("chat_history"))
    try:
        yield AGENT_START
        yield SEP_OF_LINE
        for chunk in agent_executor.stream(
            {
                "chat_history": [convert_dict_to_message(m) for m in chat_history],
                "input": user_input,
            }
        ):
            # Agent Action
            if "actions" in chunk:
                for action in chunk["actions"]:
                    yield f"Calling Tool:`{action.tool}` with input `{action.tool_input}`"
            # Observation
            elif "steps" in chunk:
                for step in chunk["steps"]:
                    yield f"Tool Result:`{step.observation}`"
            # Final result
            elif "output" in chunk:
                result = chunk["output"]
                yield f'Final Output: {chunk["output"]}'
            else:
                raise ValueError()
            yield SEP_OF_LINE
        yield AGENT_DONE
        yield result
    except Exception as e:
        channel = load_channel()
        channel.show_modal("warning", repr(e))
        logger.error(repr(e))
        yield AGENT_DONE
        yield repr(e)
        # if stream:
        #     yield from repr(e)
        # else:
        #     return repr(e)
