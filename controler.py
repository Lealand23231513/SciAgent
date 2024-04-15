import functools
import wenet
from os import system
from types import MethodType
from typing import List, Optional, Tuple, cast, Any, Union
import logging
import state
from pathlib import Path
import gradio as gr
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.adapters.openai import convert_dict_to_message
from langchain import hub
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic.v1.config import Extra
from global_var import Any, get_global_value, set_global_value
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import ToolsState
from model_state import LLMState
from channel import load_channel
from langchain_core.callbacks import (
    Callbacks,
)
from document_qa import document_qa_fn
from langchain_core.tools import StructuredTool
from websearch.arxiv_search import get_customed_arxiv_search_tool
from websearch.bing_search import get_bing_search_tool

AGENT_START = '[AGENT START]'
AGENT_DONE = '[AGENT DONE]'
SEP_OF_LINE = "".join(["-" for _ in range(8)])
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use tools only if you need it."
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

def load_custom_agent_executor(tools_inst:list[BaseTool], model_kwargs:dict[str,Any], system_prompt:Optional[str]=None):
    model = model_kwargs.pop('model')
    temperature = model_kwargs.pop('temperature')
    api_key = model_kwargs.pop('api_key')
    base_url = model_kwargs.pop('base_url')
    llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key, base_url=base_url)#type:ignore
    if len(tools_inst)==0:
        llm_with_tools = llm
    else:
        llm_with_tools = llm.bind_tools(tools_inst)
    system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
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
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[
        List[AgentAction],
        AgentFinish,
    ]:
        """Based on past history and current inputs, decide what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        inputs = {**kwargs, **{"intermediate_steps": intermediate_steps}}
        return self.runnable.invoke(inputs, config={"callbacks": callbacks})

    agent_executor = AgentExecutor(agent=agent, #type: ignore
                                   tools=tools_inst, handle_parsing_errors=True)

    agent_executor.agent.__config__.extra=Extra.allow
    agent_executor.agent.plan=MethodType(plan, agent_executor.agent)
    return agent_executor


def call_agent(user_input: str):
    # load_dotenv()
    llm_state = cast(LLMState, get_global_value("llm_state"))
    tools_state = cast(ToolsState, get_global_value("tools_state"))
    model_kwargs = llm_state.model_dump()
    if "gpt" in llm_state.model:
        agent_executor = load_openai_agent_excutor(
            tools_state.tools_inst, model_kwargs
        )
    else:
        agent_executor = load_custom_agent_executor(
            tools_state.tools_inst, model_kwargs
        )
    # chat_history = cast(list[dict[str, str]], get_global_value("chat_history"))
    try:
        yield AGENT_START
        yield SEP_OF_LINE
        for chunk in agent_executor.stream(
            {
                # "chat_history": [convert_dict_to_message(m) for m in chat_history],
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

def load_agent_executor(llm_state_name:str, tools_inst:list[BaseTool], reload_agent=False, system_prompt:Optional[str] = None):
    agent_executor = get_global_value('agent_executor')
    if agent_executor is None or reload_agent==True:
        llm_state: LLMState = get_global_value(llm_state_name)
        model_kwargs = llm_state.model_dump()
        system_prompt = system_prompt
        arxiv_search = get_customed_arxiv_search_tool()
        bing_search = get_bing_search_tool()
        agent_executor = load_custom_agent_executor(
            tools_inst+[arxiv_search, bing_search], model_kwargs, system_prompt=system_prompt
        )
        set_global_value('agent_executor', agent_executor)
    return agent_executor

def run_agent(agent_executor, user_input:str):
    try:
        yield AGENT_START
        yield SEP_OF_LINE
        for chunk in agent_executor.stream(
            {
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

def chat_with_document(filepath: str|None, user_input: str, chat_history: list, audio_path):
    state.StateMutex.set_state_mutex(True)
    reload_agent = False
    if audio_path:
        model = wenet.load_model('chinese')
        user_input += model.transcribe(audio_path)["text"]
    current_file_path = get_global_value('current_file_path')
    if current_file_path!=filepath:
        set_global_value('current_file_path', filepath)
        reload_agent = True
        logger.info("reload agent")
    chat_history.append([user_input, None])
    yield chat_history, gr.Textbox(interactive=False)
    
    if filepath:
        document_qa_tool = StructuredTool.from_function(
            func=lambda query:document_qa_fn(path=filepath, query=query),
            name="document_qa",
            description="useful for when you need to answer questions about the document.",
        )
        system_prompt = DEFAULT_SYSTEM_PROMPT + f"Help the user read the paper {Path(filepath).name}."
        # agent_executor = load_agent_executor('pdf_llm_state', [document_qa_tool, arxiv_search], reload_agent, system_prompt=system_prompt)
        agent_executor = load_agent_executor('pdf_llm_state', [document_qa_tool], reload_agent, system_prompt=system_prompt)
    else:
        agent_executor = load_agent_executor('pdf_llm_state', [], reload_agent)
    chat_history[-1][1] = ""
    generator = run_agent(agent_executor, user_input)
    for chunk in generator:
        if chunk == AGENT_DONE:
            chat_history[-1][1] += next(generator)
            yield chat_history, None
    yield chat_history, gr.Textbox(interactive=True)
    state.StateMutex.set_state_mutex(False)


# if __name__=='__main__':
#     document_qa_tool = StructuredTool.from_function(
#         func=lambda query:document_qa_fn(path='',query=query),
#         name="document_qa",
#         description="useful for when you need to answer questions about the document",
#     )

