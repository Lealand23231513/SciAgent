import openai
import json
import os
def fn_args_generator(query:str, functions, history = []):
    messages = history + [{"role": "user", "content": f"{query}"}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature = 0,
        messages=messages,
        functions=functions,
        function_call="auto",  
    )
    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        return function_args
    else:
        raise Exception("Not receive function call")
    
def translator(src:str):
    prompt = f"Please translate this sentence into English: {src}"
    messages = [{"role": "user", "content": prompt}] 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def auto_extractor(query, history = []):
    prompt = """
Extract keywords from the query:
[query]: {}
The output should be formated as below:
keyword1,keyword2,...
""".format(query)
    messages = history + [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 0,
        messages=messages  
    )
    keywords = response["choices"][0].message["content"].split(',')
    keywords = [keyword.strip() for keyword in keywords]
    return keywords

if __name__ == '__main__':
    from dotenv import load_dotenv
    
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    print(auto_extractor("What are the two components with extreme distributions that RepQ-ViT focuses on?"))