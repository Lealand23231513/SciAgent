import json
import os
import logging
from pathlib import Path
import multiprocessing

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
import requests
import urllib.request
from pathlib import Path
from urllib import parse
from scholarly import scholarly


class CustomedScholarWrapper:
    download: bool = False

    def __init__(self, **kwargs):
        super_kwargs = {}
        for k in kwargs.keys():
            if k != "download":
                super_kwargs[k] = kwargs[k]
        super().__init__(**super_kwargs)
        if kwargs.get("download"):
            self.download = True

    def run(self, query: str) -> str:

        logger = logging.getLogger(
            ".".join((Path(__file__).stem, self.__class__.__name__))
        )
        logger.info("Google Scholar search start")
        try:
            search_query = scholarly.search_pubs(query)
            results = [next(search_query) for _ in range(self.top_k_results)]
        except Exception as e:
            return f"Google Scholar exception: {e}"

        docs = []
        for result in results:
            title = result["bib"]["title"]
            url = result.get("eprint_url", "No URL available")

            if self.load_all_available_meta:
                extra_metadata = {
                    "categories": result["bib"]["fields_of_study"]
                    if "fields_of_study" in result["bib"]
                    else None,
                    "links": result.get("eprint_url", "No URL available"),
                    "Citations": result["num_citations"],
                }
            else:
                extra_metadata = {}
            metadata = {
                "Authors": ", ".join(result["bib"]["author"]),
                "Title": result["bib"]["title"],
                "Summary": result["bib"].get("abstract", "No abstract available"),
                "Published": result["bib"]["pub_year"],
                **extra_metadata,
            }
            texts = ["{}: {}".format(k, metadata[k]) for k in metadata.keys()]
            logger.info(texts)
            docs.append("\n".join(texts))

            if self.download:
                folder_name = query
                folder_name = parse.quote(folder_name, safe = "")
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                title = parse.quote(title, safe = "")
                res = requests.get(url)
                if res.status_code == 200:

                    file_name = os.path.join(folder_name, title + ".pdf")

                    with open(file_name, "wb") as pdf_file:
                        pdf_file.write(res.content)
        if docs:
            return "\n\n".join(docs)
        else:
            return "No good Google Scholar Result was found"


def get_customed_scholar_search_tool(**kwargs) -> BaseTool:
    extra_keys = [
        "top_k_results",
        "load_max_docs",
        "load_all_available_meta",
        "download",
    ]
    sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
    return CustomedScholarWrapper(**sub_kwargs)


def scholar_sarch_with_agent(user_input: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)
    tools = [
        get_customed_scholar_search_tool(load_all_available_meta=True, download=True)
    ]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    ans = agent_executor.invoke(
        {
            "input": "{}\nYou should provide reference or url link to the papers you mentioned, Published, Authors and Summary are also needed. Format your response.".format(
                user_input
            ),
        }
    )
    logger.info(ans)
    return ans["output"]
