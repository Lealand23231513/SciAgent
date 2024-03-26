import logging
import json
from pathlib import Path
from BCEmbedding.tools.langchain import BCERerank
from typing import Any, Callable, cast
from cache import Cache, load_cache
from typing import Optional, Type
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from sys import _getframe
from retrieval_const import RetrievalConst
from langchain_core.documents import Document

logger = logging.getLogger(Path(__file__).stem)


class RetrievalTool(BaseModel):
    strategy: str = RetrievalConst.DEFAULT_STRATEGY
    score_threshold: float = RetrievalConst.DEFAULT_SCORE_THRESHOLD
    k: int = RetrievalConst.DEFAULT_K


    def base_retriever(self, cache:Cache, search_type: str, search_kwargs: dict[str, Any]):
        return cache.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def BCERerank_retriever(self, cache:Cache, search_type: str, search_kwargs: dict[str, Any]):
        base_retriever = cache.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        reranker_args = {"model": "maidalun1020/bce-reranker-base_v1", "top_n": self.k}
        reranker = BCERerank(**reranker_args)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )
        return compression_retriever

    def run(self, query: str) -> str:
        logger = logging.getLogger(
            ".".join([Path(__file__).stem, _getframe().f_code.co_name])
        )
        logger.info("retrieval start")
        logger.info(self.dict())
        retireve_strategy_map: dict[str, Callable] = {
            "base": self.base_retriever,
            "BCERerank": self.BCERerank_retriever,
        }
        cache = load_cache()
        if cache is None:
            raise Exception('Cache not exist. To use retrieval tool, you need to create a cache first.')
        retriever = retireve_strategy_map[self.strategy](
            cache=cache,
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": self.score_threshold, "k":self.k},
        )
        res_docs = retriever.get_relevant_documents(query)
        res_docs = cast(list[Document], res_docs)
        result = [res_doc.dict() for res_doc in res_docs]
        logger.info(result)
        return json.dumps(result)


class RetrievalInput(BaseModel):
    """Input for the Retrieval tool."""

    query: str = Field(description="query about the cached papers")


class RetrievalQueryRun(BaseTool):
    name: str = "retrieval"
    description: str = "Retrieve the content of the cached papers and answer questions."
    retrieval_tool: RetrievalTool = Field(default_factory=RetrievalTool)
    args_schema: Type[BaseModel] = RetrievalInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Retrieval tool."""
        return self.retrieval_tool.run(query)


def get_retrieval_tool(**kwargs):
    return RetrievalQueryRun(retrieval_tool=RetrievalTool(**kwargs))


if __name__ == "main":
    tool = get_retrieval_tool()
    print(tool.schema())
