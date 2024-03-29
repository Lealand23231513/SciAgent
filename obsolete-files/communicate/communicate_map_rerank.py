# deprecated, not stable
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
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

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, response like you don't know the answer, don't try to make up an answer.
The answer should be concise and no more than 100 words.
In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:
 
question: [question here]
answer: [answer here]
score: [score between 0 and 100]
 
Begin!
 
Context:
---------
{context}
---------
question: {question}
"""

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
    #regex = '|'.join(keywords)
    #docs = [doc for doc in docs if re.search(regex, doc.page_content)]
    # if len(docs)==0:# not related query
    #     return "I can't find the answer because the question is not related to the provided papers."
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser,
    )

    chain = load_qa_chain(llm=OpenAI(temperature=0.1, max_tokens=1000, model="gpt-3.5-turbo-instruct"), 
                       chain_type="map_rerank",
                       prompt=prompt)
    ans = chain.invoke({"input_documents": docs, "question": query}, return_only_outputs=True)

    
    
    logger.info(ans)
        
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
    #test_file = r"C:\Users\15135\Documents\DCDYY\SciAgent\.cache\CLaMP(1).pdf"
    test_file = r"C:\Users\15135\Documents\DCDYY\SciAgent\.cache\CLaMP.pdf"
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    question = "What is RepQ-ViT?"
    communicate_result = communicate(test_file, question)
    print("paper: {}\nquestion: {}\nanswer: {}".format(Path(test_file),question,communicate_result))
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")