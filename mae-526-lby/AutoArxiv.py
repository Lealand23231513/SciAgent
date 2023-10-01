"""

    这个文件里有两个函数: arxiv_auto_search_and_download(top_k_results)和arxiv_auto_search(top_k_results)
    第一个函数提供搜索且下载的功能, 第二个函数仅搜索
    top_k_results参数控制返回文论文的数量
    两个函数的返回参数都是一个python字典, 都包含'title'和'arxiv_id'(字面意思)

"""
import arxiv
import requests
import openai

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema import Document

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
 
    
def arxiv_auto_search_and_download(top_k_results = 3): 
    
    openai.api_key = os.environ["OPENAI_API_KEY"]

    class ArxivAPIWrapper(BaseModel):
        
        arxiv_search: Any  #: :meta private:
        arxiv_exceptions: Any  # :meta private:
        top_k_results: int = 3 
        ARXIV_MAX_QUERY_LENGTH: int = 300
        load_max_docs: int = 100
        load_all_available_meta: bool = False
        doc_content_chars_max: Optional[int] = 40000

        @root_validator()
        def validate_environment(cls, values: Dict) -> Dict:
            """Validate that the python package exists in environment."""
            try:
                import arxiv

                values["arxiv_search"] = arxiv.Search
                values["arxiv_exceptions"] = (
                    arxiv.ArxivError,
                    arxiv.UnexpectedEmptyPageError,
                    arxiv.HTTPError,
                )
                values["arxiv_result"] = arxiv.Result
            except ImportError:
                raise ImportError(
                    "Could not import arxiv python package. "
                    "Please install it with `pip install arxiv`."
                )
            return values

        def run(self, query: str) -> str:
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
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
            except self.arxiv_exceptions as ex:
                return f"Arxiv exception: {ex}"
            docs = [
                f"Title: {result.title}\n"+ 
                f"arxiv_id: {result.entry_id[21:]}\n"
                for result in results
            ]
            if docs:
                return "\n\n".join(docs)[: self.doc_content_chars_max]
            else:
                return "No good Arxiv Result was found"

        def load(self, query: str) -> List[Document]:
            """
            Run Arxiv search and get the article texts plus the article meta information.
            See https://lukasschwab.me/arxiv.py/index.html#Search

            Returns: a list of documents with the document.page_content in text format

            Performs an arxiv search, downloads the top k results as PDFs, loads
            them as Documents, and returns them in a List.

            Args:
                query: a plaintext search query
            """  # noqa: E501
            try:
                import fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF package not found, please install it with "
                    "`pip install pymupdf`"
                )

            try:
                # Remove the ":" and "-" from the query, as they can cause search problems
                query = query.replace(":", "").replace("-", "")
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
                ).results()
            except self.arxiv_exceptions as ex:
                logger.debug("Error on arxiv: %s", ex)
                return []

            docs: List[Document] = []
            for result in results:
                try:
                    doc_file_name: str = result.download_pdf()
                    with fitz.open(doc_file_name) as doc_file:
                        text: str = "".join(page.get_text() for page in doc_file)
                except (FileNotFoundError, fitz.fitz.FileDataError) as f_ex:
                    logger.debug(f_ex)
                    continue
                if self.load_all_available_meta:
                    extra_metadata = {
                        "entry_id": result.entry_id,
                        "published_first_time": str(result.published.date()),
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                        "doi": result.doi,
                        "primary_category": result.primary_category,
                        "categories": result.categories,
                        "links": [link.href for link in result.links],
                    }
                else:
                    extra_metadata = {}
                metadata = {
                    "Published": str(result.updated.date()),
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                    "entry_id": result.entry_id,
                    **extra_metadata,
                }
                doc = Document(
                    page_content=text[: self.doc_content_chars_max], metadata=metadata
                )
                docs.append(doc)
                os.remove(doc_file_name)
            return docs

    # 引入llm模型和agent代理

    llm = ChatOpenAI(temperature=0.0)
    tools = load_tools(
        ["arxiv"], 
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    logger = logging.getLogger(__name__)

    # OpenAI的API调用函数

    def get_completion(prompt, model = "gpt-3.5-turbo"):
        messages = [{"role":"user","content":prompt}]
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = 0,
        )
        return response.choices[0].message["content"]

    # 将arxiv的返回转换成字典

    def string_to_dict(input_string):
        paragraphs = input_string.strip().split('\n\n')
        result_dict = {}
        current_dict = {}

        for paragraph in paragraphs:
            lines = paragraph.strip().split('\n')

            for line in lines:
                key, value = map(str.strip, line.split(':', 1))
                current_dict[key] = value

            if current_dict:
                result_dict[len(result_dict) + 1] = current_dict
                current_dict = {}

        return result_dict

    # 传入arxivID和论文名称并将对应论文下载到当前目录

    def download_arxiv_pdf(arxiv_id, name, folder_name):
        pdf_url = 'https://arxiv.org/pdf/' + arxiv_id + '.pdf'

        response = requests.get(pdf_url)

        if response.status_code == 200:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            filename = os.path.join(folder_name, name + '.pdf')

            with open(filename, 'wb') as pdf_file:
                pdf_file.write(response.content)
            print(f'文件 {filename} 下载成功！')
        else:
            print(f'下载失败，HTTP状态码: {response.status_code}')

    # 获取用户输入并翻译成英文

    question = input(f"您好! 我是您的论文下载助手。\n\
    请输入您想下载的论文的所属领域(单词即可，可以使用任意一种语言)，我会自动查阅在Arxiv托管的论文中的相关论文并为您下载相关度最高的{top_k_results}篇(如果有那么多的话)到当前目录：\n")
    
    prompt = f"""
    请你将该单词翻译成英文，并且只返回翻译后的英文单词：{question}
    """
    question = get_completion(prompt)

    arxiv = ArxivAPIWrapper(top_k_results = top_k_results)
    arxiv_result = arxiv.run(f"""{question}""")

    result = string_to_dict(arxiv_result)

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


    
    # 循环遍历result中的每个结果并下载论文
    for key,sub_dict in result.items():
        Title = sub_dict.get("Title").translate(translation_table) 
        download_arxiv_pdf(sub_dict.get("arxiv_id"), Title, question)
    
    return result



def arxiv_auto_search(top_k_results = 3): 
    
    openai.api_key = os.environ["OPENAI_API_KEY"]

    class ArxivAPIWrapper(BaseModel):
        
        arxiv_search: Any  #: :meta private:
        arxiv_exceptions: Any  # :meta private:
        top_k_results: int = 3 
        ARXIV_MAX_QUERY_LENGTH: int = 300
        load_max_docs: int = 100
        load_all_available_meta: bool = False
        doc_content_chars_max: Optional[int] = 40000

        @root_validator()
        def validate_environment(cls, values: Dict) -> Dict:
            """Validate that the python package exists in environment."""
            try:
                import arxiv

                values["arxiv_search"] = arxiv.Search
                values["arxiv_exceptions"] = (
                    arxiv.ArxivError,
                    arxiv.UnexpectedEmptyPageError,
                    arxiv.HTTPError,
                )
                values["arxiv_result"] = arxiv.Result
            except ImportError:
                raise ImportError(
                    "Could not import arxiv python package. "
                    "Please install it with `pip install arxiv`."
                )
            return values

        def run(self, query: str) -> str:
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
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
            except self.arxiv_exceptions as ex:
                return f"Arxiv exception: {ex}"
            docs = [
                f"Title: {result.title}\n"+ 
                f"arxiv_id: {result.entry_id[21:]}\n"
                for result in results
            ]
            if docs:
                return "\n\n".join(docs)[: self.doc_content_chars_max]
            else:
                return "No good Arxiv Result was found"

        def load(self, query: str) -> List[Document]:
            """
            Run Arxiv search and get the article texts plus the article meta information.
            See https://lukasschwab.me/arxiv.py/index.html#Search

            Returns: a list of documents with the document.page_content in text format

            Performs an arxiv search, downloads the top k results as PDFs, loads
            them as Documents, and returns them in a List.

            Args:
                query: a plaintext search query
            """  # noqa: E501
            try:
                import fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF package not found, please install it with "
                    "`pip install pymupdf`"
                )

            try:
                # Remove the ":" and "-" from the query, as they can cause search problems
                query = query.replace(":", "").replace("-", "")
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.load_max_docs
                ).results()
            except self.arxiv_exceptions as ex:
                logger.debug("Error on arxiv: %s", ex)
                return []

            docs: List[Document] = []
            for result in results:
                try:
                    doc_file_name: str = result.download_pdf()
                    with fitz.open(doc_file_name) as doc_file:
                        text: str = "".join(page.get_text() for page in doc_file)
                except (FileNotFoundError, fitz.fitz.FileDataError) as f_ex:
                    logger.debug(f_ex)
                    continue
                if self.load_all_available_meta:
                    extra_metadata = {
                        "entry_id": result.entry_id,
                        "published_first_time": str(result.published.date()),
                        "comment": result.comment,
                        "journal_ref": result.journal_ref,
                        "doi": result.doi,
                        "primary_category": result.primary_category,
                        "categories": result.categories,
                        "links": [link.href for link in result.links],
                    }
                else:
                    extra_metadata = {}
                metadata = {
                    "Published": str(result.updated.date()),
                    "Title": result.title,
                    "Authors": ", ".join(a.name for a in result.authors),
                    "entry_id": result.entry_id,
                    **extra_metadata,
                }
                doc = Document(
                    page_content=text[: self.doc_content_chars_max], metadata=metadata
                )
                docs.append(doc)
                os.remove(doc_file_name)
            return docs

    # 引入llm模型和agent代理

    llm = ChatOpenAI(temperature=0.0)
    tools = load_tools(
        ["arxiv"], 
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    logger = logging.getLogger(__name__)

    # OpenAI的API调用函数

    def get_completion(prompt, model = "gpt-3.5-turbo"):
        messages = [{"role":"user","content":prompt}]
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = 0,
        )
        return response.choices[0].message["content"]

    # 将arxiv的返回转换成字典

    def string_to_dict(input_string):
        paragraphs = input_string.strip().split('\n\n')
        result_dict = {}
        current_dict = {}

        for paragraph in paragraphs:
            lines = paragraph.strip().split('\n')

            for line in lines:
                key, value = map(str.strip, line.split(':', 1))
                current_dict[key] = value

            if current_dict:
                result_dict[len(result_dict) + 1] = current_dict
                current_dict = {}

        return result_dict

    # 获取用户输入并翻译成英文

    question = input(f"您好! 我是您的论文搜索助手。\n\
    请输入您想搜索的论文的所属领域(单词即可，可以使用任意一种语言)，我会自动查阅在Arxiv托管的论文中的相关论文并为您寻找相关度最高的{top_k_results}篇(如果有那么多的话)并以python字典的形式返回给您：\n")
    
    prompt = f"""
    请你将该单词翻译成英文，并且只返回翻译后的英文单词：{question}
    """
    question = get_completion(prompt)

    arxiv = ArxivAPIWrapper(top_k_results = top_k_results)
    arxiv_result = arxiv.run(f"""{question}""")

    result = string_to_dict(arxiv_result)
    
    return result
