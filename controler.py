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

# import env variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# description of each module
with open('modules.json', "r") as f:
    module_descriptions = json.load(f) 

user1_input = "I want to find some papers about LLM."
user2_input = "I want to write a review about LLM, and I wonder what papers can I refer to? And please write me a summary about the papers."



# define output_parser
class ModuleOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        output_format = [
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name'
                            },
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name'
                            },
                            {
                                "name" : 'module\'s name',
                                "function" : 'module\'s function name'
                            }
                        ]
        return json.dumps(output_format)
    def parse(self, text: str) :
        #TODO: Add check
        # expected_keys = [rs.name for rs in self.response_schemas]
        return json.loads(text)
output_parser = ModuleOutputParser()

# define chat_prompt
template = "You are a helpful assistant that can choose which module to execute for the user's input.\
      The modules information and function is in json format as below:\n{module_descriptions}.\
      A user will pass in a querry, and you should generate module name, function to execute in json format as output format below according to the modules information. The number of modules is probably more than 1.\
        output format:{output_format}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

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
user_input = user2_input
response = chain.run(user2_input)
# response = json.loads(response)
print(response)



#TODO: Contact controler to other modules.
