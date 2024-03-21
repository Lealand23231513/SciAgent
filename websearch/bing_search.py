from typing import Optional
from urllib import response
from langchain_core.pydantic_v1 import BaseModel, Field
from bs4 import BeautifulSoup
import requests
import logging
from langchain_text_splitters import CharacterTextSplitter
from fake_useragent import UserAgent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
logger = logging.getLogger(__name__)
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
from global_var import get_global_value
from utils import fn_args_generator
from typing import Callable, cast
from cache import Cache, load_cache
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from sys import _getframe
from functools import partial
from channel import load_channel
from tools import BaseToolState
from model_state import LLMState

prompt_template = """Use the web search results to answer the question at the end. If you don't know the answer, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""


class BingSearchConst:
    NAME= 'bing'
    DEFAULT_MAX_RESULT = 5
    MINIMUM_MAX_RESULT = 1
    MAXIMUN_MAX_RESULT = 10
    DEFAULT_TOP_K_RESULTS = 3
    MINIMUM_TOP_K_RESULTS = 1
    MAXIMUN_TOP_K_RESULTS = 5
    DEFAULT_CHUNK_SIZE = 500
    MAXIMUN_CHUNK_SIZE = 1000
    MINIMUN_CHUNK_SIZE = 500
    BASE_URL = "https://cn.bing.com/"


class BingSearchWrapper(BaseModel):
    max_results: Optional[int] = Field(
        default=BingSearchConst.DEFAULT_MAX_RESULT,
        ge=BingSearchConst.MINIMUM_MAX_RESULT,
        le=BingSearchConst.MAXIMUN_MAX_RESULT,
    )
    base_url: str = BingSearchConst.BASE_URL
    top_k_results:Optional[int] = Field(
        default=BingSearchConst.DEFAULT_TOP_K_RESULTS,
        ge=BingSearchConst.MINIMUM_TOP_K_RESULTS,
        le=BingSearchConst.MAXIMUN_TOP_K_RESULTS
    )

    @classmethod
    def _search_url(cls, query: str):
        return cls.base_url + f"search?q={query}"

    def run(self, query: str):
        """
        These code comes from https://github.com/binary-husky/gpt_academic/blob/master/crazy_functions/%E8%81%94%E7%BD%91%E7%9A%84ChatGPT_bing%E7%89%88.py
        """
        ua = UserAgent()
        headers = {"User-Agent": ua.random}
        url = self._search_url(query)
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            channel = load_channel()
            msg = repr(e)
            channel.show_modal("error", msg)
            return msg
        soup = BeautifulSoup(response.content, "html.parser")
        results = []
        for g in soup.find_all("li", class_="b_algo"):
            anchors = g.find_all("a")
            if anchors:
                link = anchors[0]["href"]
                if not link.startswith("http"):
                    continue
                title = g.find("h2").text
                item = {"title": title, "link": link}
                try:
                    text = self.scrape_text(link)
                except Exception as e:
                    channel = load_channel()
                    msg = repr(e)
                    channel.show_modal("error", msg)
                    return msg
                item["text"] = text
                results.append(item)
                logger.info(item)
                if self.max_results and len(results) == self.max_results:
                    break
        #TODO get api key in a better way
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([res['text'] for res in results])

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}
        llm_state:LLMState = get_global_value('llm_state')
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=llm_state.model, temperature=llm_state.temperature),
            chain_type="stuff",
            retriever=Chroma.from_documents(docs, OpenAIEmbeddings()).as_retriever(
                search_type="similarity_search",
                search_kwargs={'k': self.top_k_results}
            ),
            chain_type_kwargs=chain_type_kwargs,
        )
        try:
            ans = qa_chain.invoke({"query": query})
        except Exception as e:
            channel = load_channel()
            msg = repr(e)
            channel.show_modal("error", msg)
            return msg
        logger.info(ans)
        return ans["result"]
    
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
    query: str = Field(description="search query to look up")

class BingSearchQueryRun(BaseTool):
    name: str = BingSearchConst.NAME
    description:str = (
        "A wrapper around Bing."
        "Useful for when you need to find answer from the internet"
        "If you are asked some questions that you don't know the answer,"
        "you'd better try this tool."
        "Input should be a search query."
    )
    api_wrapper: BingSearchWrapper = Field(default_factory=BingSearchWrapper)
    args_schema: Type[BaseModel] = BingSearchInput

    def _run(
        self,
        query:str,
        run_manager: Optional[CallbackManagerForToolRun] = None        
    )->str:
        return self.api_wrapper.run(query)

def get_bing_search_tool(**kwargs) -> BaseTool:
    cls_properties = BingSearchWrapper.schema()['properties'].keys()
    sub_kwargs = {k: kwargs[k] for k in cls_properties if k in kwargs}
    return BingSearchQueryRun(api_wrapper=BingSearchWrapper(**sub_kwargs))
