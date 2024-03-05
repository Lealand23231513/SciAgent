from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import logging
from pathlib import Path
import os
import shutil
import json
from langchain.indexes import SQLRecordManager, index
from typing import cast
from global_var import get_global_value, set_global_value
from utils import DEFAULT_CACHE_DIR
from typing import cast
EMB_MODEL_MAP={
    "text-embedding-ada-002":{
        "api_key": None,
        "base_url": None,
    },
    "bge-m3":{
        "api_key": "EMPTY",
        "base_url": "http://10.134.40.123:6006/v1/"
    }
}
def load_cache():
    cache = cast(Cache, get_global_value('cache'))
    if cache is None:
        cache = Cache()
        set_global_value('cache',cache)
    return cache

class Cache(object):
    def __init__(self, emb_model_name:str = "text-embedding-ada-002", all_files=None, vectorstore=None, record_manager=None) -> None:
        self.emb_model_name = emb_model_name
        self.root_dir = Path(DEFAULT_CACHE_DIR)/emb_model_name
        if self.root_dir.exists() == False:
            self.root_dir.mkdir()
        self.filenames_save_path = self.root_dir/'cached-files.json'
        self.filenames_save_path.touch()
        self.cached_files_dir = self.root_dir/'cached-files'
        if self.cached_files_dir.exists()==False:
            self.cached_files_dir.mkdir()
        self.embedding = OpenAIEmbeddings(
            model = emb_model_name,
            api_key=EMB_MODEL_MAP[emb_model_name]['api_key'],
            base_url=EMB_MODEL_MAP[emb_model_name]['base_url']
        )
        #TODO: customed embedding
        if all_files is None:
            self.load_filenames()
        else:
            self.all_files = all_files
        if vectorstore is None:
            self.load_vectorstore()
        else:
            self.vectorstore = vectorstore
        if record_manager is None:
            self.load_record_manager()
        else:
            self.record_manager = record_manager
    def load_filenames(self):
        # load cached files' name
        try:
            with open(self.filenames_save_path) as f:
                self.all_files = cast(list[str], json.load(f))
        except Exception as e:
            logger = logging.getLogger(Path(__file__).stem)
            logger.error(e)
            self.all_files = []
        return self.all_files
    def load_record_manager(self):
        # load SQL record manager
        self.record_manager = SQLRecordManager(
            'chroma/'+self.emb_model_name, db_url="sqlite:///"+f"{DEFAULT_CACHE_DIR}/{self.emb_model_name}/record_manager_cache.sql"
        )
        self.record_manager.create_schema()
        return self.record_manager
    def load_vectorstore(self):
        # load Chroma vecotrstore with OpenAI embeddings
        persist_directory = str(self.root_dir/'chroma')
        self.vectorstore = Chroma(
            collection_name=self.emb_model_name, embedding_function=self.embedding, persist_directory=persist_directory
        )
        return self.vectorstore
    def save_filenames(self):
        # save cached files' name
        with open(self.filenames_save_path,'w') as f:
            json.dump(self.all_files, f)
    def clear_all(self):
        # clear all cached data
        #TODO clear cached files
        self.all_files = []
        self.save_filenames()
        for file in self.cached_files_dir.iterdir():
            file.unlink()
        logger = logging.getLogger(Path(__file__).stem)
        logger.info(f'All files in directory {DEFAULT_CACHE_DIR} are cleaned.')
        res = index([], self.record_manager, self.vectorstore, cleanup="full", source_id_key="source")
        logger.info(res)
        return res
    def cache_file(self, path:str, save_local=False, chunk_size=1000, chunk_overlap=200, add_start_index=True):
        '''
        :param path: path or url of the file
        :param save_local: whether or not save the uploaded file in local cache folder
        :param chunk_size: max length of the chunk
        '''
        if save_local == True:
            shutil.copy(path, Path(DEFAULT_CACHE_DIR+'/cached-files'))
        if(path.split(".")[-1] == 'pdf'):
            loader = PyPDFLoader(path)
        elif(path.split(".")[-1] == 'docx'):
            loader = Docx2txtLoader(path)
        else:
            logger = logging.getLogger(Path(__file__).stem)
            logger.error("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
            raise Exception("WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'." % path.split(".")[-1])
        
        raw_docs = loader.load()
        for i in range(len(raw_docs)):
            raw_docs[i].metadata['source']=Path(raw_docs[i].metadata['source']).name
        text_splitter = CharacterTextSplitter(separator='\n',chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
        docs = text_splitter.split_documents(raw_docs)
        self.all_files = set(self.all_files)# type: ignore
        self.all_files.add(Path(path).name)
        self.all_files = list(self.all_files) 
        # modify vector store
        index_res = index(
            docs,
            self.record_manager,
            self.vectorstore,
            cleanup="incremental",
            source_id_key="source"
        )
        self.save_filenames()
        logger = logging.getLogger(Path(__file__).stem)
        logger.debug(f'all files:{self.all_files}')
        logger.info(index_res)
        logger.info(f"The file ({Path(path).name}) has been cached")
        return docs
    def remove_file(self, filename:str):
        #TODO 删除某篇指定的文章
        pass
    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        newCache = Cache(
            all_files=deepcopy(self.all_files)
            )
        return newCache