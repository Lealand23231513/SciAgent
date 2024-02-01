from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from search_and_download import download_arxiv_pdf
import logging
from pathlib import Path
import os
import json
import re
import openai
from utils import fn_args_generator, auto_extractor
logger = logging.getLogger(Path(__file__).stem)

refine_prompt_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. Your answer should be concise, no more than 150 words will be best."
    "If the context isn't useful, return the original answer."
    "If you don't know the answer, response that you don't know."
)

initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n Your answer should be concise, less than 150 words is best.\n"
    "If you don't know the answer, response that you don't know."
)
def retrieve_file(path:str):
    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        logger.error("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
        raise Exception("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.info(f"The paper ({path}) has been retrieved successfully.")
    return docs



def communicate(path:str, query:str) -> str:
    '''
    :param path: path or url of the paper
    :param query: User's question about the paper  
    '''
    
    docs = retrieve_file(path)
    keywords = auto_extractor(query)
    logger.info("keywords: {}".format(', '.join(keywords)))
    regex = '|'.join(keywords)
    docs = [doc for doc in docs if re.search(regex, doc.page_content)]
    if len(docs)==0:
        return "I can't find the answer because the question is not related to the provided papers."
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )

    initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "question"], template=initial_qa_template
    )

    chain = load_qa_chain(llm=OpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo-instruct"), 
                        chain_type="refine",
                        question_prompt=initial_qa_prompt,
                        refine_prompt=refine_prompt)
    ans = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    logger.info(ans['output_text'])
        
    return ans['output_text']

def communicator_auto_runner(user_input:str, functions, history = []) -> str:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    function_args = fn_args_generator(user_input, functions, history)
    logger.debug(f"funtion args:\n{function_args}")
    path = function_args.get("path")
    query = function_args.get("query")
    result = communicate(path, query)
    return result
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    test_file = "C:/Users/15135/Documents/DCDYY/SciAgent/.cache/RepQ-ViT(1).pdf"
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    communicate_result = communicate(test_file, "What are the two components with extreme distributions that RepQ-ViT focuses on?")
    print(communicate_result)