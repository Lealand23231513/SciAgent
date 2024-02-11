# 核心控制模块
import openai
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from utils import *
from search_and_download import *
from communicate import communicator_auto_runner

logger = logging.getLogger(Path(__file__).stem)

def main(user_input:str, history, tools:list, stream:bool = False):
    # import env variables
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    user1_input = "I want to find some papers about LLM."
    user2_input = "I want to write a review about LLM, and I wonder what papers can I refer to? And please write me a summary about the papers."
    
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
            yield from exe_result
        else:
            raise Exception('exe_result is None.')

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