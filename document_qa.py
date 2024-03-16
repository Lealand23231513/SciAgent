import gradio as gr
from gradio_pdf import PDF
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import logging
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

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
    logger.info(f"The paper ({path}) has been retrieved successfully.")
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
    chain = load_qa_chain(
        llm=ChatOpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo-0125"),
        chain_type="stuff",
        prompt=prompt,
    )
    ans = chain.invoke(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    logger.info(ans)
    return ans["output_text"]

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
