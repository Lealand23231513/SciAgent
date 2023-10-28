import arxiv
import requests
import openai
import json
import os
from pathlib import Path
from urllib import parse
from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import BaseModel, root_validator
from langchain.schema import Document

class ArxivAPIWrapper(BaseModel):
    arxiv_exceptions:tuple = (
                arxiv.ArxivError,
                arxiv.UnexpectedEmptyPageError,
                arxiv.HTTPError,
            ) 
    arxiv_result:ClassVar = arxiv.Result
    top_k_results: int = 3
    ARXIV_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = 100
    load_all_available_meta: bool = False
    doc_content_chars_max: Optional[int] = 40000
    

    def run(self, query: str) -> list[dict[str, str]]:

        try:
            results = arxiv.Search(  
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results 
            ).results()
        except self.arxiv_exceptions as ex:
            print(f"Arxiv exception: {ex}")
            return []
        docs = [
            {
                "Title": result.title,
                "arxiv_id": result.entry_id[21:],
                "summary": result.summary
            }
            for result in results
        ]
        return docs

def download_arxiv_pdf(arxiv_id, file_path:Path|str):
    pdf_url = 'https://arxiv.org/pdf/' + arxiv_id + '.pdf'
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
    return response.status_code
    
    


def arxiv_auto_search_and_download(query:str, download:bool = True, top_k_results=3, folder_name = Path("./cache")) -> list[dict[str, str]]:
    """
    :return: list of arxiv results 
    [
        {
            "Tiltle":
            "arxiv_id":
            "summary":
            "path":
        },
    ]
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]

    prompt = f"""请你将该单词翻译成英文，并且只返回翻译后的英文单词：{query}"""
    messages = [{"role": "user", "content": prompt}] 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    query = response.choices[0].message["content"]

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=top_k_results)
    arxiv_result = arxiv_wrapper.run(f"""{query}""")

    
    if len(arxiv_result) == 0:
        return []

    print("get results:")
    for i,sub_dict in enumerate(arxiv_result):
        sub_dict["path"] = ""
        print(f"{i+1}.{json.dumps(sub_dict)}")

    # 如果不下载，直接返回
    if download == False:
        return arxiv_result
    

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 循环遍历result中的每个结果并下载论文
    for sub_dict in arxiv_result:
        if type(sub_dict) == dict:
            # 使用url编码，避免文件名中出现不能出现的字符
            trans_file_name = parse.quote(sub_dict["Title"]) + '.pdf'
            file_path = os.path.join(folder_name,  trans_file_name)
            status_code = download_arxiv_pdf(sub_dict["arxiv_id"], file_path)
            if status_code == 200:
                print(f'文件 {trans_file_name} 下载成功！')
                sub_dict["path"] = file_path
            else:
                print(f'文件 {trans_file_name} 下载失败！HTTP状态码: {status_code}')
                sub_dict["path"] = ""
            
    return arxiv_result

def search_and_download(user_input:str):
    messages = [{"role": "user", "content": f"{user_input}"}]
    with open('modules.json', "r") as f:
        module_descriptions = json.load(f)
    functions = module_descriptions[0]["functions"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature = 0,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    # print(response)
    response_message = response["choices"][0]["message"]
    if response_message.get("function_call"):
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args)
        arg_download = function_args.get("download")
        arg_top_k_results = function_args.get("top_k_results")
        arxiv_result = arxiv_auto_search_and_download(query = function_args.get("query"),
                                                      download=arg_download if arg_download is not None else False,
                                                      top_k_results=arg_top_k_results if arg_top_k_results is not None else 3)
        return arxiv_result
    else:
        return []
