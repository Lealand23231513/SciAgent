# 核心控制模块
import openai
from openai import OpenAI
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from search_and_download import *
from communicate import communicator_auto_runner

logger = logging.getLogger(Path(__file__).stem)

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
    
    response = chain.run(user_input)
    logger.info(response)
    return response

def main(user_input:str, history):
    global logger
    # import env variables
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    user1_input = "I want to find some papers about LLM."
    user2_input = "I want to write a review about LLM, and I wonder what papers can I refer to? And please write me a summary about the papers."
    
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f)

    response = task_decider(user_input, module_descriptions)
    exe_result = ""
    papers_info = []
    for task in response:
        functions = next(filter(lambda item: item['name'] == task["name"] ,module_descriptions))['functions']
        if task["name"] == "chatonly":
            client = OpenAI()
            messages = history + [
                {"role": "user", "content": f"{user_input}"}
            ]
            response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature = 0,
                    messages=messages
                )
            exe_result = response.choices[0].message.content
        if task["name"] == "search_and_download":
            if task["function"] == "arxiv_auto_search":
                arxiv_results = arxiv_auto_search(task["todo"], functions, history)
                for paper in arxiv_results:
                    paper["path"] = ""
                    papers_info.append(paper)
                exe_result = reporter(arxiv_results)
        if task["name"] == 'communicator':
            communicate_result = communicator_auto_runner(task['todo'], functions, history)
            exe_result = reporter(communicate_result)
        if exe_result:
            yield exe_result
        else:
            raise Exception('exe_result is None.')


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
    
def reporter(exe_result):
    '''
    Report execution result of this step to the user.
    '''
    client = OpenAI()
    messages = [
                {"role": "system", "content": "Report the full execution result to the user."},
                {"role": "assistant", "content": f"{exe_result}"}
            ]
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature = 0,
            messages=messages# type: ignore
        )
    return response.choices[0].message.content



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