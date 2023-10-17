from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate

import os
os.environ["OPENAI_API_KEY"] = "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"

def summarizer(path:str):

    if(path.split(".")[-1] == 'pdf'):
        loader = PyPDFLoader(path)
    elif(path.split(".")[-1] == 'docx'):
        loader = Docx2txtLoader(path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f'documents:{len(docs)}')

    prompt_template = """Write a summary of this paper,
    which should contain introduction of research field, process and achievements:
 
    {text}
 
    SUMMARY Here:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    llm = OpenAI(temperature=0.2, max_tokens=1000, model="gpt-3.5-turbo-instruct")
    print(llm.model_name)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
    summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]

    print("The summary here:" + summary)

summarizer("C:\Pythonfiles\langchain_try\summary\\test_paper\Attention Is All You Need.pdf")