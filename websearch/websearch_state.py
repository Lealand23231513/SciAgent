from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from websearch.const import ArxivConst, BingSearchConst, GoogleScholarConst, WebSearchStateConst

_root_dir = Path(__file__).parent.parent
import sys

sys.path.append(str(_root_dir))
from tools import BaseToolState, BaseToolkitState
from websearch.google_scholar_search import (
    get_google_scholar_search_tool,
)
from websearch.arxiv_search import get_customed_arxiv_search_tool
from websearch.bing_search import get_bing_search_tool
from config import *
from langchain_core.tools import BaseTool


class WebSearchState(BaseToolkitState):
    download: bool = WebSearchStateConst.DEFAULT_DOWNLOAD
    top_k_results: int = Field(
        default=WebSearchStateConst.DEFAULT_TOP_K_RESULTS,
        ge=WebSearchStateConst.MIN_TOP_K_RESULTS,
        le=WebSearchStateConst.MAX_TOP_K_RESULTS,
    )
    paper_search_tool_choices: list[str] = WebSearchStateConst.PAPER_SEARCH_CHOICES
    paper_search_tool_select: str = WebSearchStateConst.DEFAULT_PAPER_SEARCH_SELECT
    se_search_tool_select: str = WebSearchStateConst.DEFAULT_SE_SELECT
    se_search_tool_choices: list[str] = WebSearchStateConst.SE_CHOICES
    enable_se_search: bool = WebSearchStateConst.DEFAULT_ENABLE_SE_SEARCH

    @model_validator(mode="after")
    def validate_environ(self):
        if (
            self.paper_search_tool_select
            and self.paper_search_tool_select not in self.paper_search_tool_choices
        ):
            raise ValueError(
                f"select search subtool '{self.paper_search_tool_select}' not in available subtools {self.paper_search_tool_choices}"
            )
        return self

    @property
    def instances(self) -> list[BaseTool]:
        kwargs = self.model_dump()
        paper_search_kwargs = {
            k: kwargs[k] for k in kwargs if k in WebSearchStateConst.PAPER_SEARCH_KWARGS
        }
        paper_search_tool_map = {
            ArxivConst.NAME: get_customed_arxiv_search_tool,
            GoogleScholarConst.NAME: get_google_scholar_search_tool,
        }
        se_search_tool_map = {BingSearchConst.NAME: get_bing_search_tool}
        if self.enable_se_search:
            tools_insts = [
                paper_search_tool_map[self.paper_search_tool_select](
                    **paper_search_kwargs
                ),
                se_search_tool_map[self.se_search_tool_select](),
            ]
        else:
            tools_insts = [
                paper_search_tool_map[self.paper_search_tool_select](
                    **paper_search_kwargs
                )
            ]
        return tools_insts
