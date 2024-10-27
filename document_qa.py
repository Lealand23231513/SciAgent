from typing import Any, Literal
import gradio as gr
import os
import logging
import openai
import global_var
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from pathlib import Path
from dotenv import load_dotenv
from channel import load_channel
from model_state import LLMState
from langchain_core.tools import tool, StructuredTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.tools import BaseTool
from typing import Optional, Type, cast
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(Path(__file__).stem)


class PromptsConst(object):
    stuff_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, response like you don't know the answer, don't try to make up an answer.
    The answer should be detaild.

    context:
    ---------
    {context}
    ---------
    question: {question}
    """

    refine_initial_qa_template = """Use the following pieces of context to answer the question at the end.
    The answer should be detaild.

    context:
    {context_str}
    ---------
    question: {question}
    """

    refine_prompt_template = (
        "Refine the existing answer to the question based on the context.\n"
        "The answer should be detaild."
        "question: {question}\n"
        "------------\n"
        "exsisting answer:\n"
        "{existing_answer}\n"
        "------------\n"
        "context:\n"
        "{context_str}\n"
    )


def _get_chain(strategy: Literal["stuff", "refine"]):
    pdf_llm_state: LLMState = global_var.get_global_value("pdf_llm_state")
    llm = ChatOpenAI(
        temperature=0.1,
        model=pdf_llm_state.model,
        api_key=pdf_llm_state.api_key,  # type: ignore
        base_url=pdf_llm_state.base_url,
    )
    if strategy == "stuff":
        prompt = PromptTemplate(
            template=PromptsConst.stuff_prompt_template,
            input_variables=["context", "question"],
        )
        chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
            prompt=prompt,
        )
        return chain
    elif strategy == "refine":# 错误累积
        question_prompt = PromptTemplate(
            template=PromptsConst.refine_initial_qa_template,
            input_variables=["context_str", "question"]
        )
        refine_prompt = PromptTemplate(
            template=PromptsConst.refine_prompt_template,
            input_variables=["question","existing_answer", "context_str"],
        )

        return load_qa_chain(
            llm=llm,
            chain_type="refine",
            question_prompt=question_prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
        )
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def _load_docs(path: str):
    if path.split(".")[-1] == "pdf":
        loader = PyPDFLoader(path)
    elif path.split(".")[-1] == "docx":
        loader = Docx2txtLoader(path)
    else:
        logger.error(
            "WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'."
            % path.split(".")[-1]
        )
        raise Exception(
            "WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'."
            % path.split(".")[-1]
        )

    documents = loader.load()
    full_text = ''
    for doc in documents:
        full_text+=doc.page_content
    text_splitter = CharacterTextSplitter(separator=r"\s", is_separator_regex=True, chunk_size=12000)
    texts = text_splitter.split_text(full_text)
    docs = [Document(text) for text in texts]
    return docs

def document_qa_fn(
    path: str, query: str, strategy: Literal["stuff", "refine"] = "stuff"
) -> str:
    """
    :param path: path or url of the paper
    :param query: User's question about the paper
    """
    logger.info("document qa start")
    logger.info(f"path: {path} query: {query} strategy: {strategy}")
    docs = _load_docs(path)
    chain = _get_chain(strategy)
    try:
        ans = chain.invoke(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )
        logger.info(ans)
        return ans["output_text"]
    except Exception as e:
        channel = load_channel()
        channel.show_modal("warning", repr(e))
        logger.error(repr(e))
        return repr(e)

class DocumentQA(BaseModel):
    path:Optional[str]=None# path or url of the paper
    strategy: Literal["stuff", "refine"] = "refine"
    @root_validator()
    def validate_environment(cls, values:dict) -> dict:
        if values.get('path') is None:
            raise ValueError('path is None!')
        return values
    def run(self, question:str, method:Literal['full_text', 'retrieve']):
        self.path = cast(str, self.path)
        logger.info("document qa start")
        logger.info(f"path: {self.path} question: {question} method: {method} strategy: {self.strategy}")
        docs = _load_docs(self.path)
        chain = _get_chain(self.strategy)
        if method=='retrieve':
            db = Chroma.from_documents(docs, OpenAIEmbeddings())
            docs = db.similarity_search(question, k=5)#type:ignore
        try:
            ans = chain.invoke(
                {"input_documents": docs, "question": question}, return_only_outputs=True
            )
            logger.info(ans)
            return ans["output_text"]
        except Exception as e:
            channel = load_channel()
            channel.show_modal("warning", repr(e))
            logger.error(repr(e))
            return repr(e)

class DocumentQAInput(BaseModel):
    question: str = Field(description="用户对于文档的提问")
    method: Literal['full_text', 'retrieve'] = Field(description="阅读文档的方法，有\"full_text\"（全文阅读）和\"retrieve\"（重点阅读特定文段）两种。")

class DocumentQARun(BaseTool):
    name: str = 'document_qa'
    description: str = (
        "一个用于回答用户关于特定文档的提问的工具。如果用户提问和文章细节有关，请使用\"retrieve\"方法。"
    )
    fn: DocumentQA = Field(default_factory=DocumentQA)
    args_schema: Type[BaseModel] = DocumentQAInput

    def _run(self, question:str, method:Literal['full_text', 'retrieve'], run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        return self.fn.run(question, method)

def get_document_qa_tool(path:str):
    return DocumentQARun(fn=DocumentQA(path=path))

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    openai.api_key = os.getenv('OPENAI_API_KEY')
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    global_var._init()
    load_channel()
    from cache import init_cache
    init_cache()
    global_var.set_global_value("pdf_llm_state", LLMState())
    communicate_result = document_qa_fn(test_url, "What's the main idea of the paper?")
    print(communicate_result)
