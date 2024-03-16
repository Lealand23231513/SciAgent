import logging
from pathlib import Path
from typing import Optional, Type


from cache import load_cache
from langchain_core.tools import BaseTool
import requests
from pathlib import Path
from urllib import parse
from scholarly import scholarly
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
logger = logging.getLogger(Path(__file__).stem)

class GoogleScholarWrapper(BaseModel):
    download: bool = False
    top_k_results:int =3
    load_all_available_meta: bool = False
    load_max_docs: int = 100

    def run(self, query: str) -> str:
        def download_callback(written_path:str):
            cache = load_cache()
            cache.cache_file(written_path)
            logger.info(f'successfully download {Path(written_path).name}')
        logger = logging.getLogger(
            ".".join((Path(__file__).stem, self.__class__.__name__))
        )
        logger.info("Google Scholar search start")
        try:
            search_result_generator = scholarly.search_pubs(query)
            # results = [next(search_query) for _ in range(self.top_k_results)]
        except Exception as e:
            return f"Google Scholar exception: {e}"

        docs = []
        try:
            i=0
            while i<self.top_k_results:
                result=next(search_result_generator)
                title = result["bib"]["title"]
                url = result.get("eprint_url")
                if self.load_all_available_meta:
                    extra_metadata = {
                        "categories": result["bib"].get("fields_of_study", "No categories available"),
                        "Citations": result["num_citations"],
                    }
                else:
                    extra_metadata = {}
                metadata = {
                    "Authors": ", ".join(result["bib"]["author"]),
                    "Title": title,
                    "Summary": result["bib"].get("abstract", "No abstract available"),
                    "Published": result["bib"]["pub_year"],
                    "links": url if url else "No URL available",
                    **extra_metadata,
                }
                texts = ["{}: {}".format(k, metadata[k]) for k in metadata.keys()]
                logger.info(texts)
                docs.append("\n".join(texts))

                if self.download:
                    if url is None:
                        logger.info(f"No URL available for paper:{title}")
                        continue
                    cache = load_cache()
                    folder_name = cache.cached_files_dir
                    file_name = parse.quote(title, safe = "") + '.pdf'
                    try:
                        res = requests.get(url)
                    except Exception as e:
                        logger.exception(repr(e))
                    if res.status_code == 200:
                        file_path = Path(folder_name)/file_name
                        with open(file_path, "wb") as fp:
                            fp.write(res.content)
                        cache.cache_file(file_path)
                    else:
                        logger.info(f"Can't download paper {title} dur to a network error.\
                                    \n status code: {res.status_code}")
                i+=1
        except StopIteration:
            pass
        except Exception as e:
            return f"Meet exception during search: {repr(e)}"
        if docs:
            return "\n\n".join(docs)
        else:
            return "No good Google Scholar Result was found"

class ScholarInput(BaseModel):
    """Input for the Google Scholar wrapper tool."""

    query: str = Field(description="search query to look up")

class GoogleScholarQueryRun(BaseTool):
    """A Goolge Scholarship Wrapper runner."""
    name: str = "scholarship"
    description:str = (
        "A wrapper around Google Scholar."
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on Google Scholar. "
        "Input should be a search query."
    )
    api_wrapper: GoogleScholarWrapper = Field(default_factory=GoogleScholarWrapper)
    args_schema: Type[BaseModel] = ScholarInput

    def _run(
        self,
        query:str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    )->str:
        '''Use Google Scholar wrapper'''
        return self.api_wrapper.run(query)
            

def get_google_scholar_search_tool(**kwargs) -> BaseTool:
    cls_properties = GoogleScholarWrapper.schema()['properties'].keys()
    sub_kwargs = {k: kwargs[k] for k in cls_properties if k in kwargs.keys()}
    return GoogleScholarQueryRun(api_wrapper=GoogleScholarWrapper(**sub_kwargs))



