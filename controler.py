# 核心控制模块
from typing import Mapping
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.adapters.openai import convert_dict_to_message
from langchain import hub
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from arxiv_search import get_customed_arxiv_search_tool
from retrieval_qa import get_retrieval_tool
from global_var import set_global_value
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_zhipu import ChatZhipuAI
from functools import partial
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import load_qwen_agent_executor

logger = logging.getLogger(Path(__file__).stem)

def load_openai_agent_excutor(tools_inst:list[BaseTool], model='gpt-3.5-turbo'):
    llm = ChatOpenAI(model=model, temperature=0, api_key=os.getenv('OPENAI_API_KEY'))# type:ignore
    if len(tools_inst)==0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst,tool_choice='auto')
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
    agent_executor = AgentExecutor(agent=agent, #type: ignore
                                   tools=tools_inst, handle_parsing_errors=True)
    return agent_executor

def load_zhipuai_agent_excutor(tools_inst:list[BaseTool], model='glm-3-turbo'):
    if model=='chatglm3-6b':
        base_url = os.getenv('CHATGLM3_BASE_URL')
        api_key = "EMP.TY"
    else:
        base_url = os.getenv('ZHIPUAI_BASE_URL')
        api_key = os.getenv('ZHIPUAI_API_KEY')
    llm = ChatZhipuAI(model=model, temperature=0.01, api_key=api_key, base_url=base_url)
    if len(tools_inst)==0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst,tool_choice='auto')
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
    agent_executor = AgentExecutor(agent=agent, #type: ignore
                                   tools=tools_inst, handle_parsing_errors=True)
    return agent_executor


def call_agent(user_input:str, history:list[Mapping[str,str]], tools_choice:list, model:str, retrieval_temp:float, stream:bool = False):
    load_dotenv()
    
    tools_mapping = {
        "websearch": partial(get_customed_arxiv_search_tool, load_all_available_meta=True),
        "retrieval": get_retrieval_tool
    }
    tools_inst = [tools_mapping[tool['name']](**tool['kwargs']) for tool in tools_choice]
    agent_excutor_mapping = {
        "openai": load_openai_agent_excutor,
        "zhipuai": load_zhipuai_agent_excutor,
        "qwen": load_qwen_agent_executor,
    }
    if 'gpt' in model:
        agent_executor = agent_excutor_mapping['openai'](tools_inst, model)
    elif 'glm' in model:
        agent_executor = agent_excutor_mapping['zhipuai'](tools_inst, model)
    elif 'qwen' in model:
        agent_executor = agent_excutor_mapping['qwen'](tools_inst, model)
    set_global_value('agent_executor', agent_executor)
    ans = agent_executor.invoke(
        {
            "chat_history":[convert_dict_to_message(m) for m in history],
            "input": user_input
        }
    )
    logger.info({k:ans[k] for k in ('input', 'output')})
    if stream:
        # fake stream
        yield from ans['output']
    else:
        return ans['output']