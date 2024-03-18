from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
from utils import fn_args_generator
from typing import Callable, cast
from cache import Cache, load_cache
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from sys import _getframe
from functools import partial


logger = logging.getLogger(Path(__file__).stem)

prompt_template = """Use the following pieces of context to answer the question at the end. If you can't find answer from the context, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""

def retrieval(temperatureValue:float, query:str, path:str|None=None) -> str:
    '''
    Retrieve the content of the cached papers and response to the user's query.
    :param query: User's question about the paper
    :param path: path or url of the paper
    '''
    logger = logging.getLogger('.'.join([Path(__file__).stem, _getframe().f_code.co_name]))
    logger.info('retrieval start')
    cache = load_cache()
    if path:
        cache.cache_file(path)# TODO: check uncached files
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=temperatureValue), chain_type="stuff",# TODO: temperature, llm(changable)
                                     retriever=cache.vectorstore.as_retriever(),
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=True)
    ans = qa_chain.invoke({"query": query})
    logger.info(ans)
    return ans['result']

class RetrievalInput(BaseModel):
    """Input for the Retrieval tool."""
    query: str = Field(description="query about the cached papers")

class RetrievalQueryRun(BaseTool):
    name:str="retrieval"
    description: str=(
        "Retrieve the content of the cached papers and answer questions."
    )
    retrieve_function: Callable=Field(default=retrieval)
    args_schema: Type[BaseModel] = RetrievalInput
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Retrieval tool."""
        return self.retrieve_function(query)
def get_retrieval_tool():
    return RetrievalQueryRun(partial(retrieval, temperatureValue = 0.7))

if __name__ == "main":
    tool = get_retrieval_tool()
    tool._run()
