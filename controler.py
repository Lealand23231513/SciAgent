# 核心控制模块
import openai
import os
import json
from dotenv import load_dotenv


# import env variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# description of each module
with open('modules.json', "r") as f:
    module_descriptions = json.load(f) 


