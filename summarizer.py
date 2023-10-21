from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate

import os
import json
import openai

# DEFAULT_PATH = "./.cache"

def summary(path:str):
    '''
    :param: path: path of the file.
    '''
    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        print("document not found")
        return None
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f'documents:{len(docs)}')

    prompt_template = """Write a summary of this paper,
    which should contain introduction of research field, process and achievements:
 
    {text}
 
    SUMMARY Here:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = OpenAI(temperature=0.2, max_tokens=1000, model="gpt-3.5-turbo-instruct")
    print(llm.model_name)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
    summary_result = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
    return summary_result
def summarizer(user_input:str):
    #TODO 还没想好应该怎么处理
    return None
    messages = [{"role": "user", "content": f"{user_input}"}]
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f)
    functions = module_descriptions[1]["functions"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature = 0,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args)
        summary_result = summary(path= function_args.get("filename"))
        return summary_result
    else:
        
        return None
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "TEST  Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.pdf"
    summary_result = summary(test_file)
    print(summary_result)
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")