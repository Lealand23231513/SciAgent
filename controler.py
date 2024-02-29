# 核心控制模块
from typing import Mapping
import openai
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from Retrieval_qa import retrieval_auto_runner
from utils import *
from langchain import hub
from websearch2 import *
from websearch2 import get_customed_arxiv_search_tool
from Retrieval_qa import get_retrieval_tool
from global_var import get_global_value, set_global_value
from langchain.agents import create_openai_tools_agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.tools import tool

logger = logging.getLogger(Path(__file__).stem)

def load_openai_agent_excutor(tools_inst:list[BaseTool]):
    @tool
    def notool():
        'use no tool'
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0, api_key=os.getenv('OPENAI_API_KEY'))
    if len(tools_inst)==0:
        llm_with_tools = llm.bind_tools([notool],tool_choice='none')
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
    agent_executor = AgentExecutor(agent=agent, tools=tools_inst, handle_parsing_errors=True)#type: ignore
    return agent_executor

def call_agent(user_input:str, history:list[Mapping[str,str]], tools_choice:list, model:str, retrieval_temp:float, stream:bool = False):
    load_dotenv()
    
    tools_mapping ={
        "websearch": partial(get_customed_arxiv_search_tool, load_all_available_meta=True),
        "retrieval": get_retrieval_tool
    }
    tools_inst = [tools_mapping[tool['name']](**tool['kwargs']) for tool in tools_choice]
    # if get_global_value('tools_inst') is None or get_global_value('tools_inst')
    agent_excutor_mapping = {
        "gpt-3.5-turbo": load_openai_agent_excutor
    }
    agent_executor = agent_excutor_mapping[model](tools_inst)
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

def main(user_input:str, history, tools:list, stream:bool = False):
    '''
    deprecated
    '''
    # import env variables
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f)
    module_descriptions = list(filter(lambda item: item['name'] in tools+['chatonly'], module_descriptions))
    response = task_decider(user_input, module_descriptions)
    exe_result = ""
    papers_info = []
    for task in response:
        functions = next(filter(lambda item: item['name'] == task["name"] ,module_descriptions))['functions']
        if task["name"] == "chatonly":
            yield from chater(user_input, history, stream=stream)
            continue
        if task["name"] == "websearch":
            if task["function"] == "arxiv_auto_search":
                # try:
                #     arxiv_results = arxiv_auto_search(task["todo"], functions, history)
                #     for paper in arxiv_results:
                #         paper["path"] = ""
                #         papers_info.append(paper)
                #     exe_result = result_parser(arxiv_results, task["name"], query=user_input, stream=stream)
                # except ValueError as e:
                #     err_msg = e.args[0]
                #     exe_result = result_parser(err_msg, 'exception', query=user_input, stream=stream)
                yield from arxiv_search_with_agent(user_input=user_input)
        if task["name"] == 'retrieve':
            retrieval_result = retrieval_auto_runner(task['todo']+f"\nuser's query:\n{user_input}", functions, history)
            exe_result = retrieval_result
        yield from exe_result




def main_for_test(user_input:str):
    '''
    This is for test.
    '''
    yield "user_input:"
    yield f"{user_input}"
    yield "this"
    yield "is"
    yield "for"
    yield "test"

    #TODO: Contact controler to other modules.
if __name__ == '__main__':
    for chunk in main_for_test('TEST'):
        print(chunk)