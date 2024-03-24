from retrieval import get_retrieval_tool
from retrieval_const import RetrievalConst
from tools import BaseToolState


from pydantic import Field
from langchain_core.tools import BaseTool


class RetrievalState(BaseToolState):
    strategy: str = Field(default=RetrievalConst.DEFAULT_STRATEGY)
    score_threshold: float = Field(
        default=RetrievalConst.DEFAULT_SCORE_THRESHOLD,
        ge=RetrievalConst.MIN_SCORE_THRESHOLD,
        le=RetrievalConst.MAX_SCORE_THRESHOLD,
    )
    k: int = Field(
        default=RetrievalConst.DEFAULT_K,
        ge=RetrievalConst.MINIMUM_K,
        le=RetrievalConst.MAXIMUN_K,
    )

    @property
    def instance(self) -> BaseTool:
        kwargs = self.model_dump()
        return get_retrieval_tool(**kwargs)