from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
import os
import openai
from dotenv import load_dotenv
from utils import fn_args_generator
from typing import Callable, cast
from cache import Cache
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from sys import _getframe


logger = logging.getLogger(Path(__file__).stem)

prompt_template = """Use the following pieces of context to answer the question at the end. If you can't find answer from the context, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""

def retrieval(query:str, path:str|None=None) -> str:
    '''
    Retrieve the content of the cached papers and response to the user's query.
    :param query: User's question about the paper
    :param path: path or url of the paper
    '''
    logger = logging.getLogger('.'.join([Path(__file__).stem, _getframe().f_code.co_name]))
    logger.info('retrieval start')
    cache = Cache()
    if path:
        cache.cache_file(path)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo-0125'), chain_type="stuff",
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
    return RetrievalQueryRun()



def retrieval_auto_runner(user_input:str, functions, history = []) -> str:
    function_args = fn_args_generator(user_input, functions, history)
    logger.debug(f"funtion args:\n{function_args}")
    path = function_args.get("path")
    query = function_args.get("query")
    result = retrieval(query, path)
    return result
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    test_file = r"C:\Users\15135\Documents\DCDYY\SciAgent\.cache\cached-files\CLaMP.pdf"
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    question = "Clamp"
    communicate_result = retrieval(question, test_file)
    print("paper: {}\nquestion: {}\nanswer: {}".format(Path(test_file),question,communicate_result))