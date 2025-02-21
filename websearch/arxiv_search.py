import json
import logging
import re
import arxiv
import global_var

from pydantic.v1 import BaseModel
from pathlib import Path
from cache import load_cache
from typing import Literal, Optional, Type, Any, cast
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
    # sort_criterion:arxiv.SortCriterion = arxiv.SortCriterion.Relevance
    # sort_order:arxiv.SortOrder = arxiv.SortOrder.Descending
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

    def run(
        self,
        query: str,
        sort_criterion: arxiv.SortCriterion,
        sort_order: arxiv.SortOrder,
    ) -> str:
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
            channel = load_channel()
            if cache is None:
                msg = "请先建立知识库！"
                channel.show_modal("warning", msg)
                return
            cache.cache_file(written_path, update_ui=True)  # type:ignore
            logger.info(f"successfully download {Path(written_path).name}")

        logger.info("Arxiv search start")
        logger.info(
            f"query: {query} sort_criterion: {sort_criterion} sort_order: {sort_order}"
        )
        logger.info(self.dict())
        try:
            if self.is_arxiv_identifier(query):
                results = arxiv.Search(
                    id_list=query.split(),
                    max_results=self.top_k_results,
                    sort_by=sort_criterion,
                    sort_order=sort_order,
                ).results()
            else:
                results = arxiv.Search(
                    query[: self.ARXIV_MAX_QUERY_LENGTH],
                    max_results=self.top_k_results,
                    sort_by=sort_criterion,
                    sort_order=sort_order,
                ).results()
        except self.arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        metadatas: list[dict[str, Any]] = []
        for result in results:
            if self.download:
                msg = json.dumps(
                    {
                        "type": "funcall",
                        "name": "confirm",
                        "message": f'Do you want to download file "{result.title}" ?',
                    }
                )
                channel: Channel = global_var.get_global_value("channel")
                res = cast(str, channel.push(msg, require_response=True))
                res = json.loads(res)
                if res["response"] == True:
                    cache = load_cache()
                    if cache is None:
                        channel = load_channel()
                        msg = "请先建立知识库！"
                        channel.show_modal("warning", msg)
                        return msg
                    try:
                        filepath = result.download_pdf(
                            dirpath=str(cache.cached_files_dir)
                        )
                        download_callback(filepath)
                    except Exception as e:
                        logger.error(repr(e))
                        channel = load_channel()
                        msg = repr(e)
                        channel.show_modal("warning", msg)
                        return msg
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
            metadatas.append(metadata)
        if metadatas:
            return self.output_parser(metadatas, schema="json")
        else:
            return "No good Arxiv Result was found"

    def output_parser(
        self, raw_output: list[dict[str, Any]], schema: Literal["str", "json", "python"]
    ):
        if schema == "str":
            docs = []
            for metadata in raw_output:
                texts = ["{}: {}".format(k, metadata[k]) for k in metadata.keys()]
                docs.append("\n".join(texts))
            logger.info(docs)
            return "\n\n".join(docs)[: self.doc_content_chars_max]
        elif schema == "python":
            logger.info(raw_output)
            return repr(raw_output)
        elif schema == "json":
            output = json.dumps(raw_output)
            logger.info(output)
            return output
        else:
            raise ValueError(f"Error schema: {schema}")


# below comes from langchain_community.utilities.arxiv.tools
class ArxivInput(BaseModel):
    """Input for the Arxiv tool."""

    query: str = Field(description="search query to look up")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        description="sort criterion"
    )
    sort_order: Literal["ascending", "descending"] = Field(description="search order")


class CustomArxivQueryRun(BaseTool):
    """Tool that searches the Arxiv API."""

    name: str = ArxivConst.NAME
    description: str = (
        "A wrapper around Arxiv.org "
        "Use this tool only if you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Note that this tool cannot be used for searching news."
        "Input should be a search query."
    )
    api_wrapper: CustomArxivAPIWrapper = Field(default_factory=CustomArxivAPIWrapper)
    args_schema: Type[BaseModel] = ArxivInput

    def _run(
        self,
        query: str,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"],
        sort_order: Literal["ascending", "descending"],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Arxiv tool."""
        sort_by_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        sort_order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        return self.api_wrapper.run(
            query, sort_by_map[sort_by], sort_order_map[sort_order]
        )


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
