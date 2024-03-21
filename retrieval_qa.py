from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
from global_var import get_global_value
from utils import fn_args_generator
from typing import Callable, cast
from cache import Cache, load_cache
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from sys import _getframe
from functools import partial
from channel import load_channel
from tools import BaseToolState
from model_state import LLMState


logger = logging.getLogger(Path(__file__).stem)

prompt_template = """Use the following pieces of context to answer the question at the end. If you can't find answer from the context, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""


class RetrievalStateConst:
    MAX_TEMPERATURE = 0.99
    MIN_TEMPERATURE = 0.01
    DEFAULT_TEMPERATURE = 0.5
    MAX_TOP_P = 0.99
    MIN_TOP_P = 0.01
    DEFAULT_TOP_P = 0.7
    MAX_SCORE_THRESHOLD = 1
    MIN_SCORE_THRESHOLD = 0
    DEFAULT_SCORE_THRESHOLD = 0.1



class RetrievalState(BaseToolState):
    temperature: float = Field(
        default=RetrievalStateConst.DEFAULT_TEMPERATURE,
        gt=RetrievalStateConst.MIN_TEMPERATURE,
        lt=RetrievalStateConst.MAX_TEMPERATURE,
    )
    top_p: float = Field(
        default=RetrievalStateConst.DEFAULT_TOP_P,
        gt=RetrievalStateConst.MIN_TOP_P,
        lt=RetrievalStateConst.MAX_TOP_P,
    )    
    score_threshold: float = Field(
        default=RetrievalStateConst.DEFAULT_SCORE_THRESHOLD,
        ge=RetrievalStateConst.MIN_SCORE_THRESHOLD,
        le=RetrievalStateConst.MAX_SCORE_THRESHOLD,
    )

    @property
    def instance(self) -> BaseTool:
        return get_retrieval_tool()


def retrieval(
    query: str, 
    temperature: float = RetrievalStateConst.DEFAULT_TEMPERATURE,
    top_p:float = RetrievalStateConst.DEFAULT_TOP_P,
    score_threshold:float = RetrievalStateConst.DEFAULT_SCORE_THRESHOLD
) -> str:
    """
    Retrieve the content of the cached papers and response to the user's query.
    :param query: User's question about the paper
    """
    logger = logging.getLogger(
        ".".join([Path(__file__).stem, _getframe().f_code.co_name])
    )
    logger.info("retrieval start")
    cache = load_cache()
    if cache is None:
        channel = load_channel()
        msg = "请先建立知识库！"
        channel.show_modal("error", msg)
        return msg
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": prompt}
    llm_state:LLMState = get_global_value('llm_state')
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=llm_state.model, temperature=temperature, model_kwargs={"top_p": top_p}),
        chain_type="stuff",  # TODO: temperature, llm(changable)
        retriever=cache.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': score_threshold}
        ),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    try:
        ans = qa_chain.invoke({"query": query})
    except Exception as e:
        channel = load_channel()
        msg = repr(e)
        channel.show_modal("error", msg)
        return msg
    logger.info(ans)
    return ans["result"]


class RetrievalInput(BaseModel):
    """Input for the Retrieval tool."""

    query: str = Field(description="query about the cached papers")


class RetrievalQueryRun(BaseTool):
    name: str = "retrieval"
    description: str = "Retrieve the content of the cached papers and answer questions."
    retrieve_function: Callable = Field(default=retrieval)
    args_schema: Type[BaseModel] = RetrievalInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Retrieval tool."""
        return self.retrieve_function(query)


def get_retrieval_tool(**kwargs):
    return RetrievalQueryRun(retrieve_function=partial(retrieval,**kwargs))


if __name__ == "main":
    tool = get_retrieval_tool()
    print(tool.schema())
    # tool._run()
