from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
import os
from langchain_community.adapters.openai import convert_dict_to_message
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
import logging
from pathlib import Path
from pydantic.v1.config import Extra
from pydantic import BaseModel, model_validator, root_validator, validator
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.adapters.openai import convert_dict_to_message
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
import logging
from pathlib import Path
from typing import (
    Any,
    List,
    Tuple,
    Union,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    Callbacks,
)
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.tools import BaseTool
from langchain.chains.llm import LLMChain
from types import MethodType
from global_var import *
logger = logging.getLogger(Path(__file__).stem)



def chater(query:str, history, stream=False, api_key:str|None=None):
    messages = history + [{"role": "user", "content": f"{query}"}]
    messages = [convert_dict_to_message(m) for m in messages]
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125',streaming=stream, api_key=api_key)#type: ignore
    for chunk in llm.stream(messages):
        yield chunk.content

# define output_parser
class ModuleOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        output_format = [
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name',
                                "todo": 'Assuming you are the user, write a query telling the model what you want to do with this function'
                            }
                        ]
        return json.dumps(output_format)
    def parse(self, text: str) -> list:
        return json.loads(text)

def task_decider(user_input:str, module_descriptions): 
    
    # define chat prompt
    system_template = "You are a helpful assistant that can choose which module to execute for the user's input.\
        The modules' information and function is in json format as below:\n{module_descriptions}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "You should generate the modules' name, function to execute and the function's target based on the user's input.\
        The user's input is as below:\n{text}\n \
        The output should be formatted as json. The format requirement is as below:\n{output_format}\n\
        Attention: The numer of the modules can be more than 1."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # define output parser
    output_parser = ModuleOutputParser()

    chat_prompt =  ChatPromptTemplate(
        messages=[system_message_prompt, human_message_prompt],
        input_variables=["text"],
        partial_variables={
            "module_descriptions": str(module_descriptions),
            "output_format": output_parser.get_format_instructions()
            }
    )

    chain = LLMChain(
        llm=ChatOpenAI(temperature=0),
        prompt=chat_prompt,
        output_parser=output_parser
    )
    
    response = chain.invoke({"text": user_input})
    logger.info(response['text'])
    return response['text']


def result_parser(raw_exe_result, exe_module:str, query:str|None=None, stream=False, api_key:str|None=None):
    '''
    Parse execution result of execution to the user.
    '''
    system_msg = 'You are a useful assistant that can summary, induction, extract information.'
    messages = []
    if exe_module == 'websearch':
        if query is None:
            raise Exception('query is None and exe_module is websearch!')
        user_msg = f"Reply to the user's input according to the information that is in JSON array format and contained some paper's metadata. You should provide reference to the papers you mentioned. Answer the question directly without using any expression that is similar to \"According to the JSON array\" or \"In the given JSON array\" in your reply.\nJSON array:\n{raw_exe_result}\n\nuser's input:\n{query}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    elif exe_module == 'retrieve':
        raise Exception('retrieve does not need a result_parser')
    elif exe_module == 'exception':
        user_msg = f"Reply to the user's input according to the information. Answer directly without using any expression such as \"accoring to the information\" or \"in the information\".\ninformation:\n{raw_exe_result}\nuser\'s input:\n{query}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    
    messages = [convert_dict_to_message(m) for m in messages]
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125',streaming=stream, api_key=api_key)#type: ignore
    for chunk in llm.stream(messages):
        yield chunk.content

def judger(history, question):
    client = OpenAI()
    messages = history + [
        {"role": "user", "content": f"Make a \"True\" or \"False\" decision about this question based on historical information:{question}\n Answer the question simply by \"True\" or \"False\""}
    ]
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature = 0,
            messages=messages
        )
    if response.choices[0].message.content not in ("True", "False"):
        raise Exception("judger: response not in (\"True\", \"False\")")
    return response.choices[0].message.content

def fn_args_generator(query:str, functions, history = []):
    client = OpenAI()
    messages = history + [{"role": "user", "content": f"{query}"}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature = 0,
        messages=messages,
        functions=functions,
        function_call="auto",  
    )
    response_message = response.choices[0].message
    logger.debug(response_message)
    if response_message.function_call:
        function_args = json.loads(response_message.function_call.arguments)
        return function_args  
    else:
        raise Exception("Not receive function call")
    
def translator(src:str):
    client = OpenAI()
    prompt = f"Please translate this sentence into English: {src}"
    messages = [{"role": "user", "content": prompt}] 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages, # type: ignore
        temperature=0,
    )
    return response.choices[0].message.content

def auto_extractor(query, history = []):
    client = OpenAI()
    prompt = """
Extract keywords from the query:
[query]: {}
The output should be formated as below:
keyword1,keyword2,...
""".format(query)
    messages = history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature = 0,
        messages=messages  
    )
    keywords = response.choices[0].message.content
    if keywords:
        keywords = keywords.split(',')
        keywords = [keyword.strip() for keyword in keywords]
    else:
        raise Exception('response.choices[0].message.content is None')
    return keywords

def load_qwen_agent_executor(tools_inst:list[BaseTool], model_kwargs:dict[str,Any]):
    model = model_kwargs.pop('model')
    temperature = model_kwargs.pop('temperature')
    api_key = model_kwargs.pop('api_key')
    base_url = model_kwargs.pop('base_url')
    llm = ChatOpenAI(model=model, temperature=temperature,api_key=api_key,base_url=base_url, model_kwargs=model_kwargs)
    llm = ChatOpenAI(model=model, temperature=0, api_key="EMPTY", base_url=base_url)#type:ignore
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


if __name__ == '__main__':
    from dotenv import load_dotenv
    import openai
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(auto_extractor("What are the two components with extreme distributions that RepQ-ViT focuses on?"))