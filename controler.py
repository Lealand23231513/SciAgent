# 核心控制模块
import openai
import os
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from search_and_download import search_and_download
from summarizer import summarizer

# define output_parser
class ModuleOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        output_format = [
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name',
                            },
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name',
                            },
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name',
                            }
                        ]
        return json.dumps(output_format)
    def parse(self, text: str) -> list:
        print(text)
        return json.loads(text)

def main(user_input:str):
    # import env variables
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # description of each module
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f) 

    user1_input = "I want to find some papers about LLM."
    user2_input = "I want to write a review about LLM, and I wonder what papers can I refer to? And please write me a summary about the papers."

    # user_input = user2_input
    

    # define chat prompt
    system_template = "You are a helpful assistant that can choose which module to execute for the user's input.\
        The modules information and function is in json format as below:\n{module_descriptions}.\
        A user will pass in a querry, and you should generate module name, function to execute in json format as output format below according to the modules information. The number of modules is probably more than 1.\
            output format:{output_format}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{text}"
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
    for mission in response:
        if mission["name"] == "search_and_download":
            response_message = search_and_download(user_input)
            print(response_message)
            yield response_message
        # elif mission["name"] == "summarizer":
        #     response_message = summarizer()

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
