from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

_root_dir = Path(__file__).parent.parent
import sys

sys.path.append(str(_root_dir))
from tools import BaseToolState
from websearch.google_scholar_search import get_google_scholar_search_tool, GoogleScholarConst
from websearch.arxiv_search import get_customed_arxiv_search_tool, ArxivConst
from config import *
from langchain_core.tools import BaseTool


class WebSearchStateConst:
    DEFAULT_DOWNLOAD = False
    DEFAULT_TOP_K_RESULTS = 3
    DEFAULT_LOAD_ALL_AVAILABLE_META = False
    DEFAULT_LOAD_MAX_DOCS = 100
    PAPER_SEARCH_CHOICES = [ArxivConst.NAME, GoogleScholarConst.NAME]
    DEFAULT_PAPER_SEARCH_SELECT = PAPER_SEARCH_CHOICES[0]
    DEFAULT_TOP_K_RESULTS = 3
    MAX_TOP_K_RESULTS = 10
    MIN_TOP_K_RESULTS = 1
    PAPER_SEARCH_KWARGS = ["download", "top_k_results"]


class WebSearchState(BaseToolState):
    download: bool = WebSearchStateConst.DEFAULT_DOWNLOAD
    top_k_results: int = Field(
        default=WebSearchStateConst.DEFAULT_TOP_K_RESULTS,
        ge=WebSearchStateConst.MIN_TOP_K_RESULTS,
        le=WebSearchStateConst.MAX_TOP_K_RESULTS,
    )
    sub_tool_choices: list[str] = WebSearchStateConst.PAPER_SEARCH_CHOICES
    sub_tool_select: str = WebSearchStateConst.DEFAULT_PAPER_SEARCH_SELECT

    @model_validator(mode="after")
    def validate_environ(self):
        if self.sub_tool_select and self.sub_tool_select not in self.sub_tool_choices:
            raise ValueError(
                f"select search subtool '{self.sub_tool_select}' not in available subtools {self.sub_tool_choices}"
            )
        return self

    @property
    def instance(self) -> BaseTool:
        kwargs = self.model_dump()
        search_kwargs = {
            k: kwargs[k] for k in kwargs if k in WebSearchStateConst.PAPER_SEARCH_KWARGS
        }
        sub_tool_map = {
            ArxivConst.NAME: get_customed_arxiv_search_tool,
            GoogleScholarConst.NAME: get_google_scholar_search_tool,
        }

        return sub_tool_map[self.sub_tool_select](**search_kwargs)
