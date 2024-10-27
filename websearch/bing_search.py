import multiprocessing.pool
import random
import requests
import time
import logging
import os
import json
import multiprocessing 
from typing import Optional, Any
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from channel import load_channel
from websearch.const import BingSearchConst


logger = logging.getLogger(__name__)

prompt_template = """Use the web search results to answer the question at the end. If you don't know the answer, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""


class BingSearchWrapper(BaseModel):
    bing_subscription_key: Optional[str] = None
    '''bing search api key, by default, use os.environ['BING_SUBSCRIPTION_KEY']'''
    max_results: int = Field(
        default=BingSearchConst.DEFAULT_MAX_RESULT,
        ge=BingSearchConst.MINIMUM_MAX_RESULT,
        le=BingSearchConst.MAXIMUN_MAX_RESULT,
    )
    '''
    max search webpages
    '''
    bing_search_url: str = BingSearchConst.bing_search_url
    top_k_results: int = Field(
        default=BingSearchConst.DEFAULT_TOP_K_RESULTS,
        ge=BingSearchConst.MINIMUM_TOP_K_RESULTS,
        le=BingSearchConst.MAXIMUN_TOP_K_RESULTS,
    )
    '''max search result to return'''
    chunk_size: int = Field(
        default=BingSearchConst.DEFAULT_CHUNK_SIZE,
        ge=BingSearchConst.MINIMUN_CHUNK_SIZE,
        le=BingSearchConst.MAXIMUN_CHUNK_SIZE,
    )
    chunk_overlap: int = Field(
        default=BingSearchConst.DEFAULT_CHUNK_OVERLAP,
        ge=BingSearchConst.MIN_CHUNK_OVERLAP,
        le=BingSearchConst.MAX_CHUNK_OVERLAP,
    )
    max_retries:int = BingSearchConst.DEFAULT_MAX_RETRIES
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""

    @root_validator(pre=True)
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and endpoint exists in environment."""
        if values.get('bing_subscription_key') == None:
            values["bing_subscription_key"] = os.getenv("BING_SUBSCRIPTION_KEY")
        return values

    # def _search_url(self, query: str):
    #     return self.bing_search_url + f"search?q={query}&form=ANNTH1&pc=U531&mkt=zh-CN"

    def _get_results_from_query(self, query:str, count:int):
        # def format_result(search_result:dict):
        #     text = self.add_text(search_result['url'])
        #     search_result["text"] = text
        #     return search_result
        
        headers = {"Ocp-Apim-Subscription-Key": self.bing_subscription_key}
        params = {
            "q": query,
            "count": count,
            "textDecorations": True,
            "textFormat": "HTML",
            **self.search_kwargs,
        }
        response = requests.get(
            self.bing_search_url,
            headers=headers,
            params=params,  # type: ignore
        )
        response.raise_for_status()
        search_results = response.json()
        results:list[dict[str,Any]] = []
        if "webPages" in search_results:
            pool = multiprocessing.Pool()
            raw_results = []
            # for search_result in search_results["webPages"]["value"]:
            #     item = {"title": search_result['name'], "link": search_result['url']}
            #     raw_results.append(item)
            raw_results = [{"title": search_result['name'], "link": search_result['url']} for search_result in search_results["webPages"]["value"]]
            try:
                print(raw_results)
                results = pool.map(self.add_text, raw_results)
            except Exception as e:
                print(repr(e))
            # for search_result in search_results["webPages"]["value"]:
            #     item = {"title": search_result['name'], "link": search_result['url']}
            #     text = self.scrape_text(search_result['url'])
            #     item["text"] = text
            #     results.append(item)
        logger.info(results)
        return results

    def run(self, query: str):
        """
        These code comes from https://github.com/binary-husky/gpt_academic/blob/master/crazy_functions/%E8%81%94%E7%BD%91%E7%9A%84ChatGPT_bing%E7%89%88.py
        """
        logger.info("bing search start")
        logger.info({"query": query})

        try:
            results = self._get_results_from_query(query, count=self.max_results)
        except Exception as e:
            channel = load_channel()
            msg = repr(e)
            channel.show_modal("warning", msg)
            return msg
        if len(results)==0:
            return 'No related results found from bing'
        
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=self.chunk_size,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        metadatas = []
        for res in results:
            metadatas.append({k:res[k] for k in res if k!='text'})
        docs = text_splitter.create_documents(texts = [res["text"] for res in results], metadatas=metadatas)

        db = Chroma.from_documents(docs, OpenAIEmbeddings())
        res_docs = db.similarity_search(query,k=self.top_k_results)#type:ignore
        logger.info(res_docs)
        return json.dumps([doc.dict() for doc in res_docs], ensure_ascii=False)

    def add_text(self, search_result:dict) -> dict[str,Any]:
        """
        These code comes from https://github.com/binary-husky/gpt_academic/blob/master/crazy_functions/%E8%81%94%E7%BD%91%E7%9A%84ChatGPT_bing%E7%89%88.py
        """
        """Scrape text from a webpage

        Args:
            url (str): The URL to scrape text from

        Returns:
            str: The scraped text
        """
        from fake_useragent import UserAgent
        url = search_result['link']
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            "Content-Type": "text/plain",
        }
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.encoding == "ISO-8859-1":
                response.encoding = response.apparent_encoding
            time.sleep(max(random.random()*BingSearchConst.MAX_TIME_SLEEP, 0.5))
        except Exception as e:
            raise e
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        search_result['text'] = text
        return search_result


class BingSearchInput(BaseModel):
    query: str = Field(description="search query to look up, better in Chinese")


class BingSearchQueryRun(BaseTool):
    name: str = BingSearchConst.NAME
    description: str = (
        "A wrapper around Bing."
        "Useful for when you need to find answer from the internet"
        "If you are asked some questions that you don't know the answer,"
        "you'd better try this tool."
        "Input should be a search query, better in Chinese."
        "Output is the search results."
        "If you are using search results for answering questions, be sure to give sources and links"
    )
    api_wrapper: BingSearchWrapper = Field(default_factory=BingSearchWrapper)
    args_schema: Type[BaseModel] = BingSearchInput
    '''bing search api key, by default, use os.environ['BING_SUBSCRIPTION_KEY']'''

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Any:
        return self.api_wrapper.run(query)


def get_bing_search_tool(**kwargs) -> BaseTool:
    cls_properties = BingSearchWrapper.schema()["properties"].keys()
    sub_kwargs = {k: kwargs[k] for k in cls_properties if k in kwargs}
    return BingSearchQueryRun(api_wrapper=BingSearchWrapper(**sub_kwargs))
