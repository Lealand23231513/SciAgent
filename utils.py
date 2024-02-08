from openai import OpenAI
import json
import os
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

if __name__ == '__main__':
    from dotenv import load_dotenv
    import openai
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(auto_extractor("What are the two components with extreme distributions that RepQ-ViT focuses on?"))