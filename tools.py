from arxiv_search import get_customed_arxiv_search_tool
from retrieval_qa import get_retrieval_tool
from google_scholar_search import get_google_scholar_search_tool
from functools import partial
from config import *

TOOLS_MAPPING = {
    "websearch": get_google_scholar_search_tool,
    "retrieval": get_retrieval_tool
}

# def validate_tools(tools, tools_kwargs):
    

# def load_tools(tools, tools_kwargs):
    