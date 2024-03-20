import gradio as gr
import os
import logging
import openai
import global_var
from gradio_pdf import PDF
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from pathlib import Path
from dotenv import load_dotenv
from channel import load_channel
from model_state import LLMState

logger = logging.getLogger(Path(__file__).stem)


prompt_template_stuff = """Use the following pieces of context to answer the question at the end. If you don't know the answer, response like you don't know the answer, don't try to make up an answer.
The answer should be detaild.

context:
---------
{context}
---------
question: {question}
"""


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
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def document_qa_fn(path: str, query: str) -> str:
    """
    :param path: path or url of the paper
    :param query: User's question about the paper
    """
    logger.info("File QA start")
    docs = _load_docs(path)
    prompt = PromptTemplate(
        template=prompt_template_stuff,
        input_variables=["context", "question"],
    )
    pdf_llm_state: LLMState = global_var.get_global_value("pdf_llm_state")
    llm = ChatOpenAI(
        temperature=0.1, 
        model=pdf_llm_state.model, 
        api_key=pdf_llm_state.api_key, # type: ignore
        base_url=pdf_llm_state.base_url
    )  
    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
    )
    try:
        ans = chain.invoke(
            {"input_documents": docs, "question": query}, return_only_outputs=True
        )
        logger.info(ans)
        return ans["output_text"]
    except Exception as e:
        channel = load_channel()
        channel.show_modal("error", repr(e))
        logger.error(repr(e))
        return repr(e)


if __name__ == "__main__":
    dir = Path("document_qa_examples")

    def fn(question: str, filepath: str) -> str:
        output = document_qa_fn(filepath, question)
        return output

    demo = gr.Interface(
        fn,
        [
            gr.Textbox(label="Question"),
            PDF(height=1000, label="Document", min_width=300),
        ],
        gr.Textbox(),
        examples=[
            ["Who is the first writer of the paper?", str(dir / "example1.pdf")],
            [
                "What is the biggest advantage of the new model?",
                str(dir / "example2.pdf"),
            ],
        ],
    )
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    demo.launch(inbrowser=True)
