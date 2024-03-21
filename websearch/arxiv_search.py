import json
import logging
import re
import arxiv
import global_var
import multiprocessing

from pydantic.v1 import BaseModel
from pathlib import Path
from cache import load_cache
from typing import Optional, Type, Any, cast
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from functools import partial
from channel import Channel, load_channel
from websearch.const import ArxivConst, WebSearchStateConst

logger = logging.getLogger(__name__)


class CustomArxivAPIWrapper(BaseModel):
    """
    We write this wrapper based on langchain arxiv api wrapper: langchain_community.utilities.arxiv.ArxivAPIWrapper
    """

    arxiv_search: Any = arxiv.Search
    arxiv_exceptions: Any = (
        arxiv.ArxivError,
        arxiv.UnexpectedEmptyPageError,
        arxiv.HTTPError,
    )
    top_k_results: int = WebSearchStateConst.DEFAULT_TOP_K_RESULTS
    ARXIV_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = WebSearchStateConst.DEFAULT_LOAD_MAX_DOCS
    load_all_available_meta: bool = WebSearchStateConst.DEFAULT_LOAD_ALL_AVAILABLE_META
    doc_content_chars_max: Optional[int] = 4000
    download: bool = WebSearchStateConst.DEFAULT_DOWNLOAD

    def is_arxiv_identifier(self, query: str) -> bool:
        """Check if a query is an arxiv identifier."""
        arxiv_identifier_pattern = r"\d{2}(0[1-9]|1[0-2])\.\d{4,5}(v\d+|)|\d{7}.*"
        for query_item in query[: self.ARXIV_MAX_QUERY_LENGTH].split():
            match_result = re.match(arxiv_identifier_pattern, query_item)
            if not match_result:
                return False
            assert match_result is not None
            if not match_result.group(0) == query_item:
                return False
        return True

    def run(self, query: str) -> str:
        """
        We overwrite ArxivAPIWrapper.run() to fit into our framework
        Args:
            query: a plaintext search query
        """
        logger = logging.getLogger(
            ".".join((Path(__file__).stem, self.__class__.__name__))
        )

        def download_callback(written_path: str):
            cache = load_cache()
            if cache is None:
                channel = load_channel()
                msg = "请先建立知识库！"
                channel.show_modal("error", msg)
            cache.cache_file(written_path)  # type:ignore
            logger.info(f"successfully download {Path(written_path).name}")

        logger.info("Arxiv search start")
        logger.info(f"query: {query}")
        try:
            if self.is_arxiv_identifier(query):
                results = arxiv.Search(
                    id_list=query.split(),
                    max_results=self.top_k_results,
                ).results()
            else:
                results = arxiv.Search(
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
                        "message": f'Do you want to download file "{result.title}" ?',
                    }
                )
                channel = cast(Channel, global_var.get_global_value("channel"))
                res = cast(str, channel.push(msg, require_response=True))
                res = json.loads(res)
                if res["response"] == True:
                    cache = load_cache()
                    if cache is None:
                        channel = load_channel()
                        msg = "请先建立知识库！"
                        channel.show_modal("error", msg)
                        return msg
                    pool.apply_async(
                        partial(
                            result.download_pdf, dirpath=str(cache.cached_files_dir)
                        ),
                        callback=download_callback,
                        error_callback=lambda err: logger.error(err),
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
                }
            else:
                extra_metadata = {}
            metadata = {
                "Published": str(result.updated.date()),
                "Title": result.title,
                "Authors": ", ".join(a.name for a in result.authors),
                "Summary": result.summary,
                "Links": [link.href for link in result.links],
                **extra_metadata,
            }
            texts = ["{}: {}".format(k, metadata[k]) for k in metadata.keys()]
            logger.info(texts)
            docs.append("\n".join(texts))
        pool.close()
        pool.join()
        if docs:
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"


# below comes from langchain_community.utilities.arxiv.tools
class ArxivInput(BaseModel):
    """Input for the Arxiv tool."""

    query: str = Field(description="search query to look up")


class CustomArxivQueryRun(BaseTool):
    """Tool that searches the Arxiv API."""

    name: str = ArxivConst.NAME
    description: str = (
        "A wrapper around Arxiv.org "
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Input should be a search query."
    )
    api_wrapper: CustomArxivAPIWrapper = Field(default_factory=CustomArxivAPIWrapper)
    args_schema: Type[BaseModel] = ArxivInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Arxiv tool."""
        return self.api_wrapper.run(query)


def get_customed_arxiv_search_tool(**kwargs) -> BaseTool:
    extra_keys = [
        "top_k_results",
        "load_max_docs",
        "load_all_available_meta",
        "download",
    ]
    sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs.keys()}
    return CustomArxivQueryRun(api_wrapper=CustomArxivAPIWrapper(**sub_kwargs))


def arxiv_search_with_agent(user_input: str):
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)
    tools = [
        get_customed_arxiv_search_tool(load_all_available_meta=True, download=False)
    ]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)  # type: ignore
    ans = agent_executor.invoke({"input": user_input})
    logger.info(ans)
    return ans["output"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    arxiv_search_with_agent("AI")
