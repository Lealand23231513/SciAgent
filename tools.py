import abc
from arxiv_search import get_customed_arxiv_search_tool
from global_var import get_global_value
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

class WebSearchStateConst():
    DEFAULT_DOWNLOAD = False
    DEFAULT_TOP_K_RESULTS = 3
    DEFAULT_LOAD_ALL_AVAILABLE_META = False
    DEFAULT_LOAD_MAX_DOCS = 100

class WebSearchState(BaseToolState):
    download: bool = WebSearchStateConst.DEFAULT_DOWNLOAD

    @property
    def instance(self) -> BaseTool:
        kwargs = self.model_dump()
        return get_google_scholar_search_tool(**kwargs)

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