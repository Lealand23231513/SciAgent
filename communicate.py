from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from search_and_download import download_arxiv_pdf

import os
os.environ["OPENAI_API_KEY"] = ""

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
)

initial_qa_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n Your answer should be concise, less than 150 words is best.\n"
)

def communicate(path:str):

    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)
    else:
        print("论文文件格式错误")
        os._exit(0)
    print("The paper has been uploaded successfully.")

    print("Please wait for a moment......")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

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
    
    query = input("Ask question to SciAgent? (If nothing to ask, please press 'n' or 'N'):")

    while query != 'N' and query != 'n':
        ans = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        print(ans['output_text'])
        query = input("What else do you want to ask? (If nothing to ask, please press 'n' or 'N'):")
    
    print("Thank you for using! Hoping these answers above are helpful to you.")

    return 

def summarizer(papers_info):
    ai_response = []
    for i,paper_info in enumerate(papers_info):
        file_path = download_arxiv_pdf(paper_info)
        papers_info[i]["path"] = file_path
        #communicate_result = communicate(file_path, "question")
        ai_response.append(f"Succesfully download <{paper_info['Title']}> into {file_path} !\n The summary result is as below:\n{summary_result}")
    return "\n".join(ai_response)
    
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "TEST  Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series.pdf"
    #communicate_result = communicate(file_path, "question")
    print(summary_result)
    #summary("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")