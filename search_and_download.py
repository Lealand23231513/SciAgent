from typing_extensions import Unpack
import arxiv
from pydantic.config import ConfigDict
import requests
import openai
import json
import os
import logging
from pathlib import Path
from urllib import parse
from enum import Enum
from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import BaseModel, root_validator
from langchain.schema import Document

logger = logging.getLogger(Path(__file__).stem)
from arxiv.arxiv import SortCriterion, SortOrder
from utils import fn_args_generator


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
    sort_by:SortCriterion = SortCriterion.Relevance
    sort_order:SortOrder = SortOrder.Ascending
    

    def run(self, query:str,sort_by:SortCriterion, sort_order:SortOrder) -> list[dict[str, str]]:

        try:
            results = arxiv.Search(  
                query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results, sort_by=self.sort_by, sort_order=self.sort_order
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


def arxiv_auto_search(user_input:str, functions, history = [], sort_by:SortCriterion = SortCriterion.Relevance, sort_order:SortOrder = SortOrder.Ascending) -> list[dict[str, str]]:
    """
    :return: list of arxiv results 
    [
        {
            "Title":
            "arxiv_id":
            "summary":
            "url":
        },
    ]
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]

    function_args = fn_args_generator(user_input, functions, history)    

    top_k_results = function_args.get("top_k_results")
    if top_k_results:
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=top_k_results,sort_by=sort_by,sort_order=sort_order)
    else:
        arxiv_wrapper = ArxivAPIWrapper()
    query = function_args.get("query")

    arxiv_result = arxiv_wrapper.run(f"""{query}""",sort_by,sort_order)
    
    if len(arxiv_result) == 0:
        raise Exception("No arxiv result found")

    logger.info("get results:")
    for i,sub_dict in enumerate(arxiv_result):
        sub_dict["url"] = 'https://arxiv.org/pdf/' + sub_dict["arxiv_id"] + '.pdf'
        logger.info(f"{i+1}.{json.dumps(sub_dict.get('Title'))}")
        logger.debug(f"{i+1}.{json.dumps(sub_dict)}")

    return arxiv_result

def download_arxiv_pdf(paper_info, folder_name = Path("./.cache"), replace_exist = False):
    trans_file_name = parse.quote(paper_info["Title"], safe='') + '.pdf'
    file_path = os.path.abspath(os.path.join(folder_name,  trans_file_name))
    pdf_url = paper_info["url"]
    response = requests.get(pdf_url)
    response.raise_for_status() # raise HTTPError

    # 如果文件已经存在，抛出异常，需要用户决定是否下载更换文件
    #TODO if replace_exist == False and os.path.exists(file_path):
    #     raise Exception(f"Path {file_path} already exist, do you want to replace it?")
        
    with open(file_path, 'wb') as pdf_file:
        pdf_file.write(response.content)
    return file_path
