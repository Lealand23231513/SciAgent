"""
    这个文件里有两个函数: arxiv_auto_search_and_download(top_k_results)和arxiv_auto_search(top_k_results)
    第一个函数提供搜索且下载的功能, 第二个函数仅搜索
    top_k_results参数控制返回文论文的数量
    两个函数的返回参数都是一个python字典, 都包含'title'和'arxiv_id'(字面意思)
"""
import arxiv
import requests
import openai
import json
import os
from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import BaseModel, root_validator
from langchain.schema import Document


#TODO:将翻译集成到ArxivAPIWrapper中,或者集成到一个chain里
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

    def run(self, query: str) -> list[dict[str, str]] | str:
        """
        Performs an arxiv search and A single string
        with the publish date, title, authors, and summary
        for each article separated by two newlines.

        If an error occurs or no documents found, error text
        is returned instead. Wrapper for
        https://lukasschwab.me/arxiv.py/index.html#Search

        Args:
            query: a plaintext search query
        """  # noqa: E501
        try:
            results = arxiv.Search(  
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
            ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            {
                "Title": result.title,
                "arxiv_id": result.entry_id[21:]
            }
            for result in results
        ]

        if docs:
            return docs
        else:
            return "No good Arxiv Result was found"

def download_arxiv_pdf(arxiv_id, ori_name:str, folder_name:str):
    pdf_url = 'https://arxiv.org/pdf/' + arxiv_id + '.pdf'
    response = requests.get(pdf_url)

    if response.status_code == 200:
        # 定义一个字符映射表并创建翻译表，用来替换文件名中不能出现的字符
        char_mapping = {
            '\\': ' ',
            '/': ' ',
            '?': ' ',
            ':': ' ',
            '<': ' ',
            '>': ' ',
            '|': ' ',
            '*': ' ',
            '"': ' ',
        }
        translation_table = str.maketrans(char_mapping)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        filename = os.path.join(folder_name.translate(translation_table), ori_name.translate(translation_table) + '.pdf')

        with open(filename, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f'文件 {filename} 下载成功！')
    else:
        print(f'下载失败，HTTP状态码: {response.status_code}')

# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#     )
#     return response.choices[0].message["content"]

def arxiv_auto_search_and_download(query:str, download:bool = False, top_k_results:int=3) -> list[dict[str, str]] | str | None:

    openai.api_key = os.environ["OPENAI_API_KEY"]

    # 获取用户输入并翻译成英文

    # query = input(f"您好! 我是您的论文下载助手。\n\
    # 请输入您想下载的论文的所属领域(单词即可，可以使用任意一种语言)，我会自动查阅在Arxiv托管的论文中的相关论文并为您下载相关度最高的{top_k_results}篇(如果有那么多的话)到当前目录：\n")

    # prompt = f"""
    # 请你将该单词翻译成英文，并且只返回翻译后的英文单词：{query}
    # """
    # question = get_completion(query)

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=top_k_results)
    arxiv_result = arxiv_wrapper.run(f"""{query}""")

    if type(arxiv_result) == str:
        print(arxiv_result)
        return None
    
    
    print("get results:")
    for i,sub_dict in enumerate(arxiv_result):
        print(f"{i+1}.{json.dumps(sub_dict)}")

    # 如果不下载，直接返回
    if download == False:
        return arxiv_result

    # 循环遍历result中的每个结果并下载论文
    for sub_dict in arxiv_result:
        if type(sub_dict) == dict:
            download_arxiv_pdf(sub_dict["arxiv_id"], sub_dict["Title"], query)

    return arxiv_result
