from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.output_parsers import RegexParser
from search_and_download import download_arxiv_pdf
import logging
from pathlib import Path
import os
import json
import openai
from utils import fn_args_generator
logger = logging.getLogger(Path(__file__).stem)

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (\d*)",
    output_keys=["answer", "score"],
    default_output_key="answer"
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
 
In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:
 
Question: [question here]
Helpful Answer In Italian: [answer here]
Score: [score between 0 and 100]
 
Begin!
 
Context:
---------
{context}
---------
Question: {question}
Concise Answer no more than 100 words here:"""

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
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser,
    )

    chain = load_qa_chain(llm=OpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo-instruct"), 
                        chain_type="map_rerank",
                        prompt=prompt)
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
    test_file = "C://Users//15135//Documents//DCDYY//PLLaMa.pdf"
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    communicate_result = communicate(test_url, "what's the main idea of this article?")
    # print(summary_result)
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")