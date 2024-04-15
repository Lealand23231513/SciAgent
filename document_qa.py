from typing import Literal
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
