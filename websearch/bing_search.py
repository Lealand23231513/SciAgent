import random
import requests
import time
import logging
from typing import Optional, Any
from langchain_core.pydantic_v1 import BaseModel, Field
from bs4 import BeautifulSoup
from langchain_text_splitters import CharacterTextSplitter
from fake_useragent import UserAgent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path
from global_var import get_global_value
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from channel import load_channel
from model_state import LLMState
from websearch.const import BingSearchConst, WebSearchStateConst

logger = logging.getLogger(__name__)

prompt_template = """Use the web search results to answer the question at the end. If you don't know the answer, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""


class BingSearchWrapper(BaseModel):
    max_results: Optional[int] = Field(
        default=BingSearchConst.DEFAULT_MAX_RESULT,
        ge=BingSearchConst.MINIMUM_MAX_RESULT,
        le=BingSearchConst.MAXIMUN_MAX_RESULT,
    )
    base_url: str = BingSearchConst.BASE_URL
    top_k_results: Optional[int] = Field(
        default=BingSearchConst.DEFAULT_TOP_K_RESULTS,
        ge=BingSearchConst.MINIMUM_TOP_K_RESULTS,
        le=BingSearchConst.MAXIMUN_TOP_K_RESULTS,
    )
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

    def _search_url(self, query: str):
        return self.base_url + f"search?q={query}&form=ANNTH1&pc=U531"
    
    def _get_results_from_query(self, query:str):
        results:list[dict[str,Any]] = []
        for i in range(self.max_retries+1):
            ua = UserAgent()
            headers = {"User-Agent": ua.edge}
            url = self._search_url(query)
            response = requests.get(url, headers=headers)
            time.sleep(max(random.random()*BingSearchConst.MAX_TIME_SLEEP, 0.5))
            soup = BeautifulSoup(response.content, "html.parser")
            for g in soup.find_all("li", class_="b_algo"):
                anchors = g.find_all("a")
                if anchors:
                    link = anchors[0]["href"]
                    # print(link)
                    if not link.startswith("http"):
                        continue
                    title = g.find("h2").text
                    item = {"title": title, "link": link}
                    try:
                        search_text = self.scrape_text(link)
                    except Exception as e:
                        raise e
                        
                    item["text"] = search_text
                    results.append(item)
                    if self.max_results and len(results) == self.max_results:
                        break
            logger.info({"try": i, "results": results})
            if results:
                break
        return results

    def run(self, query: str):
        """
        These code comes from https://github.com/binary-husky/gpt_academic/blob/master/crazy_functions/%E8%81%94%E7%BD%91%E7%9A%84ChatGPT_bing%E7%89%88.py
        """
        logger.info("bing search start")
        logger.info({"query": query})

        try:
            results = self._get_results_from_query(query)
        except Exception as e:
            channel = load_channel()
            msg = repr(e)
            channel.show_modal("error", msg)
            return msg
        if len(results)==0:
            return 'No related results found from bing'
        # TODO get api key in a better way
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
        res_docs = db.similarity_search(query,k=self.top_k_results)
        logger.info(res_docs)
        search_text_lst = []
        for res_doc in res_docs:
            metadata = res_doc.metadata
            search_text = '\n'.join(f"{k}: {metadata[k]}" for k in metadata)
            search_text = search_text + '\n' + f'text: {res_doc.page_content}'
            search_text_lst.append(search_text)
        return '\n'.join(search_text_lst)

    def scrape_text(self, url: str) -> str:
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
        return text


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
        "If you are using search results for answering questions, remember to provide link to the souce of the results."
    )
    api_wrapper: BingSearchWrapper = Field(default_factory=BingSearchWrapper)
    args_schema: Type[BaseModel] = BingSearchInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return self.api_wrapper.run(query)


def get_bing_search_tool(**kwargs) -> BaseTool:
    cls_properties = BingSearchWrapper.schema()["properties"].keys()
    sub_kwargs = {k: kwargs[k] for k in cls_properties if k in kwargs}
    return BingSearchQueryRun(api_wrapper=BingSearchWrapper(**sub_kwargs))
