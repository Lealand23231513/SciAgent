import abc
from arxiv_search import get_customed_arxiv_search_tool
from global_var import get_global_value
from retrieval_qa import get_retrieval_tool
from google_scholar_search import get_google_scholar_search_tool
from functools import partial
from pydantic import BaseModel, model_validator, Field
from typing import Any, Literal, Optional, cast, Callable, Optional
from enum import Enum
from config import *
from state import BaseState
from langchain_core.tools import BaseTool

class BaseToolState(BaseState, abc.ABC):

    @property
    @abc.abstractmethod
    def instance(self) -> BaseTool:
        """The instance of the BaseTool."""


class WebSearchState(BaseToolState):
    download: bool = False

    @property
    def instance(self) -> BaseTool:
        kwargs = self.model_dump()
        return get_google_scholar_search_tool(**kwargs)


class RetrievalConst:
    MAX_TEMPERATURE = 1
    MIN_TEMPERATURE = 0
    DEFAULT_TEMPERATURE = 0.5
    MAX_TOP_P = 1
    MIN_TOP_P = 0
    DEFAULT_TOP_P = 0.7
    MAX_CHUNK_SIZE = 4000
    MIN_CHUNK_SIZE = 500
    DEFAULT_CHUNK_SIZE = 1000
    MAX_SCORE_THRESHOLD = 1
    MIN_SCORE_THRESHOLD = 0
    DEFAULT_SCORE_THRESHOLD = 0.1
    MAX_CHUNK_OVERLAP = 200
    MIN_CHUNK_OVERLAP = 200
    DEFAULT_CHUNK_OVERLAP = 200


class RetrievalState(BaseToolState):
    temperature: float = Field(
        default=RetrievalConst.DEFAULT_TEMPERATURE,
        ge=RetrievalConst.MIN_TEMPERATURE,
        le=RetrievalConst.MAX_TEMPERATURE,
    )
    top_p: float = Field(
        default=RetrievalConst.DEFAULT_TOP_P,
        ge=RetrievalConst.MIN_TOP_P,
        le=RetrievalConst.MAX_TOP_P,
    )
    chunk_size: int = Field(
        default=RetrievalConst.DEFAULT_CHUNK_SIZE,
        ge=RetrievalConst.MIN_CHUNK_SIZE,
        le=RetrievalConst.MAX_CHUNK_SIZE,
    )
    score_threshold: float = Field(
        default=RetrievalConst.DEFAULT_SCORE_THRESHOLD,
        ge=RetrievalConst.MIN_SCORE_THRESHOLD,
        le=RetrievalConst.MAX_SCORE_THRESHOLD,
    )
    chunk_overlap: int = Field(
        default=RetrievalConst.DEFAULT_CHUNK_OVERLAP,
        ge=RetrievalConst.MIN_CHUNK_OVERLAP,
        le=RetrievalConst.MAX_CHUNK_OVERLAP,
    )

    @property
    def instance(self) -> BaseTool:
        return get_retrieval_tool()

class ToolsState(BaseState):
    tools_select: list[str] = []
    tools_choices: list[str] = SUPPORT_TOOLS

    @model_validator(mode="after")
    def validate_environ(cls, values):
        for tool_select in values.tools_select:
            if tool_select not in SUPPORT_TOOLS:
                raise ValueError(
                    f"tools {values.tools_select} not in SUPPORT_TOOLS {SUPPORT_TOOLS}"
                )
        return values

    @property
    def tools_inst(self) -> list[BaseTool]:
        tools_inst_lst: list[BaseTool] = []
        for tool in self.tools_select:
            tool_state_name = self.name2key(tool)
            tool_state = cast(BaseToolState, get_global_value(tool_state_name))
            tools_inst_lst.append(tool_state.instance)
        return tools_inst_lst
    
    def name2key(self,tool_name:str)->str:
        return tool_name+'_state'