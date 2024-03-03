import json
import os
import logging
from pathlib import Path
import multiprocessing
from cache import load_cache
logger = logging.getLogger(Path(__file__).stem)
from utils import DEFAULT_CACHE_DIR
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import BaseTool
from functools import partial
import global_var
from typing import cast
from channel import Channel


class CustomedArxivAPIWrapper(ArxivAPIWrapper):
    download:bool=False
    def __init__(self, **kwargs):
        super_kwargs = {}
        for k in kwargs.keys():
            if k != "download":
                super_kwargs[k]=kwargs[k]
        super().__init__(**super_kwargs)
        if kwargs.get('download'):
            self.download = True
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
        logger = logging.getLogger('.'.join((Path(__file__).stem, self.__class__.__name__)))
        def download_callback(written_path):
            cache = load_cache()
            cache.cache_file(written_path)
            logger.info(f'successfully download {Path(written_path).name}')
        
        logger.info('Arxiv search start')
        try:
            if self.is_arxiv_identifier(query):
                results = self.arxiv_search(
                    id_list=query.split(),
                    max_results=self.top_k_results,
                ).results()
            else:
                results = self.arxiv_search(  # type: ignore
                    query[: self.ARXIV_MAX_QUERY_LENGTH], max_results=self.top_k_results
                ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = []
        pool = multiprocessing.Pool()
        for result in results:
            if self.download:
                msg = json.dumps(
                    {
                        "type": "funcall",
                        "name": "confirm",
                        "message": f"Do you want to download file \"{result.title}\" ?"
                    }
                ) 
                channel = cast(Channel, global_var.get_global_value('channel'))
                res = cast(str, channel.push(msg,require_response=True))
                res = json.loads(res)
                if res['response'] == True:
                    pool.apply_async(
                        partial(result.download_pdf, dirpath=DEFAULT_CACHE_DIR+'/cached-files'), 
                        callback=download_callback, 
                        error_callback=lambda err:logger.error(err)
                    )
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
                "Summary": result.summary,
                **extra_metadata,
            }
            texts = ['{}: {}'.format(k, metadata[k]) for k in metadata.keys()]
            logger.info(texts)
            docs.append('\n'.join(texts))
        pool.close()
        pool.join()
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

def get_customed_arxiv_search_tool(**kwargs) -> BaseTool:
    extra_keys = ["top_k_results", "load_max_docs", "load_all_available_meta", "download"]
    sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
    return ArxivQueryRun(api_wrapper=CustomedArxivAPIWrapper(**sub_kwargs))

def arxiv_search_with_agent(user_input:str):
    llm = ChatOpenAI(model='gpt-3.5-turbo-0125',temperature=0.5)
    tools = [get_customed_arxiv_search_tool(load_all_available_meta=True, download=True)]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)#type: ignore
    ans = agent_executor.invoke(
        {
            "input": "{}\nYou should provide reference or url link to the papers you mentioned, Published, Authors and Summary are also needed. Format your response.".format(user_input),
        }
    )
    logger.info(ans)
    return ans['output']
