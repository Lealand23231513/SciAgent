from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.prompts import PromptTemplate

import os
import sys
os.environ["OPENAI_API_KEY"] = 

path = input("输入你要进行概括的论文路径（支持pdf和word版）:")
if(path.split(".")[-1] == 'pdf'):
    loader = PyPDFLoader(path)
elif(path.split(".")[-1] == 'docx'):
    loader = Docx2txtLoader(path)
else:
    print("论文文件格式错误")
    sys.exit()

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
print(f'documents:{len(docs)}')

prompt_template = """Write a summary of this paper,
which should contain introduction of research field, process and achievements:
 
{text}
 
SUMMARY IN Chinese:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = OpenAI(temperature=0.2, max_tokens=1000)
chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
summary = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
print(summary)