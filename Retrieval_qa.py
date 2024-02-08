from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
import os
import json
import openai
import re
from utils import fn_args_generator, auto_extractor
logger = logging.getLogger(Path(__file__).stem)
embeddings = OpenAIEmbeddings()

output_parser = RegexParser(
    regex=r"answer: (.*?)\nscore: (\d*)",
    output_keys=["answer", "score"],
    default_output_key="answer"
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you can't find answer from the context, you should response like you can't find answer from the context, don't try to make up an answer.
Context:
---------
{context}
---------
question: {question}
"""

def retrieve_file(path:str, chunk_size=1000, chunk_overlap=200, add_start_index=True):
    '''
    :param path: path or url of the paper
    :param chunk_size: max length of the chunk
    '''
    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        logger.error("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
        raise Exception("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n',chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
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
    vec_store = Chroma.from_documents(docs, embeddings)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser,
    )

    chain_type_kwargs = {"prompt": prompt}
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo-0125'), chain_type="stuff",
                                     retriever=vec_store.as_retriever(),
                                     chain_type_kwargs=chain_type_kwargs,
                                     return_source_documents=True)
    ans = qa_chain.invoke({"query": query})
    print(ans)
    logger.info(ans['result'])
        
    return ans['result']

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
    test_file = r"C:\Users\15135\Documents\DCDYY\SciAgent\.cache\CLaMP.pdf"
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    question = "What is RepQ-ViT?"
    communicate_result = communicate(test_file, question)
    print("paper: {}\nquestion: {}\nanswer: {}".format(Path(test_file),question,communicate_result))