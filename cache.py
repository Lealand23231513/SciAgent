import logging
import shutil
import json
import atexit
import pickle
from model_state import EMBState, EMBStateConst
from venv import logger
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
import logging
import copy
from pathlib import Path

from langchain.indexes import SQLRecordManager, index
from typing import Any, Literal, cast

from librosa import cache
from global_var import get_global_value, set_global_value
from config import DEFAULT_CACHE_DIR, EMB_MODEL_MAP
from typing import cast, List, Optional
from state import BaseState
from channel import load_channel
from pydantic import Field

class CacheConst():
    DEFAULT_CACHE_DIR=Path(DEFAULT_CACHE_DIR)
    LAST_RUNCACHE_CONFIG_PATH =DEFAULT_CACHE_DIR/"lastrun-settings.json"
    CACHE_LIST_PATH=DEFAULT_CACHE_DIR/"cache_list.json"
    EMB_MODEL_MAP=EMB_MODEL_MAP
    MAX_CHUNK_SIZE = 4000
    MIN_CHUNK_SIZE = 500
    DEFAULT_CHUNK_SIZE = 1000
    MAX_CHUNK_OVERLAP = 200
    MIN_CHUNK_OVERLAP = 50
    DEFAULT_CHUNK_OVERLAP = 200

class CacheState(BaseState):
    chunk_size: int = Field(
        default=CacheConst.DEFAULT_CHUNK_SIZE,
        ge=CacheConst.MIN_CHUNK_SIZE,
        le=CacheConst.MAX_CHUNK_SIZE,
    )
    chunk_overlap: int = Field(
        default=CacheConst.DEFAULT_CHUNK_OVERLAP,
        ge=CacheConst.MIN_CHUNK_OVERLAP,
        le=CacheConst.MAX_CHUNK_OVERLAP,
    )

def _clear_last_run_cache(**last_run_cache_config):
    cache=Cache(**last_run_cache_config)
    cache.clear_all()

def _write_last_run_cache_config(last_run_cache_config:dict[str,str]):
    with open(CacheConst.LAST_RUNCACHE_CONFIG_PATH, "w") as f:
        json.dump(last_run_cache_config, f)

def _write_cache_lst():
    cache_lst = get_global_value('cache_lst')
    with open(CacheConst.CACHE_LIST_PATH,'w') as f:
        json.dump(cache_lst,f)

def _clear_non_utf8_ch(docs):
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.encode(errors='ignore').decode()
    return docs
class Cache(object):
    def __init__(
        self,
        namespace: str,
        emb_model_name: str,
        emb_api_key:Optional[str]=None,
        emb_base_url:Optional[str]=None
    ) -> None:
        self.emb_model_name = emb_model_name
        self.namespace = namespace
        self.root_dir = CacheConst.DEFAULT_CACHE_DIR / namespace
        atexit.register(self._prepare_del)
        if self.root_dir.exists() == False:
            self.root_dir.mkdir(parents=True)
        self.cache_config_path = self.root_dir / "config.json"
        self.allfiles_lst_path = self.root_dir / "cached-files.json"
        if self.allfiles_lst_path.exists() == False:
            self.allfiles_lst_path.touch()
            with open(self.allfiles_lst_path,'w') as f:
                json.dump([],f)
        self.cached_files_dir = self.root_dir / "cached-files"
        self.config = {
            "namespace": namespace,
            "emb_model_name": emb_model_name
        }
        if self.cached_files_dir.exists() == False:
            self.cached_files_dir.mkdir(parents=True)
        self.emb_save_path = self.root_dir/'emb.pkl'
        
        if self.emb_save_path.exists() and emb_api_key is None and emb_base_url is None:
            with open(self.emb_save_path,'rb') as f:
                self.emb_config:EMBState = pickle.load(f)
        else:
            self.emb_config = EMBState(model=emb_model_name, api_key=emb_api_key, base_url=emb_base_url)
        if self.emb_config.model=='bce-embbedding-base_v1':#TODO Complete this part
            embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}
            from huggingface_hub import login
            login(self.emb_config.api_key)
            self.embedding = HuggingFaceEmbeddings(
                model_name='maidalun1020/bce-embedding-base_v1',
                encode_kwargs=embedding_encode_kwargs
            )
        else:
            self.embedding = OpenAIEmbeddings(
                model=self.emb_config.model,
                api_key=self.emb_config.api_key,#type:ignore #CacheConst.EMB_MODEL_MAP[emb_model_name]["api_key"],
                base_url=self.emb_config.base_url#CacheConst.EMB_MODEL_MAP[emb_model_name]["base_url"],
            )
        self.load_filenames()
        self.load_vectorstore()
        self.load_record_manager()

        # save configs
        self.save_config()
        self.save_emb()
        cache_lst = cast(list[str], get_global_value('cache_lst'))
        if self.namespace not in cache_lst:
            cache_lst.append(self.namespace)
            _write_cache_lst()
    
    def save_config(self):
        with open(self.cache_config_path,'w') as f:
            json.dump(self.config,f)
    def save_allfiles_lst(self):
        # save cached files' name
        with open(self.allfiles_lst_path, "w") as f:
            json.dump(self.all_files, f)
    def save_emb(self):
        with open(self.emb_save_path, 'wb') as f:
            pickle.dump(self.emb_config,f)

    def load_filenames(self):
        # load cached files' name
        try:
            with open(self.allfiles_lst_path) as f:
                self.all_files = cast(list[str], json.load(f))
        except Exception as e:
            logger = logging.getLogger(Path(__file__).stem+'.load_filenames')
            logger.info(repr(e))
            self.all_files = []
        return self.all_files

    def load_record_manager(self):
        # load SQL record manager
        self.record_manager = SQLRecordManager(
            self.namespace + self.emb_model_name,
            db_url="sqlite:///"
            + f"{str(CacheConst.DEFAULT_CACHE_DIR)}/{self.namespace}/record_manager_cache.sql",
        )
        self.record_manager.create_schema()
        return self.record_manager

    def load_vectorstore(self):
        # load Chroma vecotrstore with OpenAI embeddings
        persist_directory = str(self.root_dir / "chroma")
        self.vectorstore = Chroma(
            embedding_function=self.embedding,
            persist_directory=persist_directory,
        )
        return self.vectorstore

    def clear_all(self):
        for file in self.cached_files_dir.iterdir():
            file.unlink()
        logger = logging.getLogger(Path(__file__).stem)
        logger.info(f"All files in directory {self.root_dir} are cleaned.")
        res = index(
            [],
            self.record_manager,
            self.vectorstore,
            cleanup="full",
            source_id_key="source",
        )
        logger.info(res)
        self.all_files = []
        self.save_allfiles_lst()

    def cache_file(
        self,
        path: str|Path,
        save_local=False,
        add_start_index=False,
        update_ui=False
    ):
        """
        :param path: path or url of the file
        :param save_local: whether or not save the uploaded file in local cache folder
        :param chunk_size: max length of the chunk
        """
        if save_local == True:
            shutil.copy(path, self.cached_files_dir)
        if isinstance(path, str)==False:
            path = str(path)
        path = cast(str,path)
        if path.split(".")[-1] == "pdf":
            loader = PyPDFLoader(path)
        elif path.split(".")[-1] == "docx":
            loader = Docx2txtLoader(path)
        else:
            logger = logging.getLogger(Path(__file__).stem)
            logger.error(
                "WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'."
                % path.split(".")[-1]
            )
            raise Exception(
                "WRONG EXTENSION: expect '.pdf' or '.docx', but receive '%s'."
                % path.split(".")[-1]
            )

        raw_docs = loader.load()
        raw_docs = _clear_non_utf8_ch(raw_docs)
        for i in range(len(raw_docs)):
            raw_docs[i].metadata["source"] = Path(raw_docs[i].metadata["source"]).name
        cache_state:CacheState = get_global_value('cache_state')
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=cache_state.chunk_size,
            chunk_overlap=cache_state.chunk_overlap,
            add_start_index=add_start_index,
        )
        docs = text_splitter.split_documents(raw_docs)
        new_filename = Path(path).name
        if new_filename not in self.all_files:
            self.all_files.append(new_filename)
        # modify vector store
        index_res = index(
            docs,
            self.record_manager,
            self.vectorstore,
            cleanup="incremental",
            source_id_key="source",
        )
        self.save_allfiles_lst()
        logger = logging.getLogger(Path(__file__).stem)
        logger.debug(f"all files:{self.all_files}")
        logger.info(index_res)
        logger.info(f"The file ({Path(path).name}) has been cached")
        if update_ui:
            channel = load_channel()
            channel.send_reload()
            channel.show_modal('info',f"文件 {Path(path).name} 上传成功!")
        return index_res

    def delete_file(self, filename: str):
        logger = logging.getLogger(Path(__file__).stem)
        uids_to_delete = self.record_manager.list_keys(group_ids=[filename])
        num_deleted = 0
        if uids_to_delete:
            self.vectorstore.delete(uids_to_delete)
            self.record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)
            logger.info(f"The file ({filename}) has been deleted")
        delete_res = {
            "num_added": 0,
            "num_updated": 0,
            "num_skipped": 0,
            "num_deleted": num_deleted,
        }
        self.all_files=[f for f in self.all_files if f!=filename]
        self.save_allfiles_lst()
        logger.info(delete_res)
    def delete_cache(self):
        '''
        delete cache in default cache dir
        '''
        self.clear_all()
        self.emb_save_path.unlink()
        self.cache_config_path.unlink()
        self.allfiles_lst_path.unlink()
        cache_lst = get_global_value('cache_lst')
        cache_lst = [item for item in cache_lst if item!=self.namespace]
        set_global_value('cache_lst',cache_lst)
        _write_cache_lst()
        logger.info(f'delete cache {self.namespace}')
    def _prepare_del(self):
        self.save_allfiles_lst()
    def __del__(self):
        pass

def init_cache(clear_old:bool=False, **kwargs):
    '''
    :param clear_old: if True, will delete old cache if last run cache config is different from given kwargs
    :param kwargs: kwargs of Cache
    :param create_only: if True, only create a new cache, not change current cache config
    '''
    logger = logging.getLogger(Path(__file__).stem+'init_cache')
    if CacheConst.DEFAULT_CACHE_DIR.exists() == False:
        CacheConst.DEFAULT_CACHE_DIR.mkdir(parents=True)
    if CacheConst.CACHE_LIST_PATH.exists()==False:
        CacheConst.CACHE_LIST_PATH.touch()
        with open(CacheConst.CACHE_LIST_PATH,'w') as f:
            json.dump([],f)
    with open(CacheConst.CACHE_LIST_PATH) as f:
        try:
            cache_lst = cast(list[str], json.load(f))
        except Exception as e:
            logger.error(repr(e))
            cache_lst=[]
    set_global_value('cache_lst', cache_lst)
    if CacheConst.LAST_RUNCACHE_CONFIG_PATH.exists():
        try:
            with open(CacheConst.LAST_RUNCACHE_CONFIG_PATH) as f:
                last_run_cache_config = cast(dict[str, Any], json.load(f))
        except Exception as e:
            last_run_cache_config = {}
        if last_run_cache_config!=kwargs:
            if clear_old:
                logger.info(f'last run cache config {last_run_cache_config} is different from given kwargs, so clear last run cache and build a new cache')
                _clear_last_run_cache(**last_run_cache_config)
            last_run_cache_config.update(kwargs)
        new_cache_config:dict[str,Any] = last_run_cache_config   
    else:
        logger.info("Can't find last run config, will create a config file")
        CacheConst.LAST_RUNCACHE_CONFIG_PATH.touch()
        new_cache_config = {}
    _write_last_run_cache_config(new_cache_config)
    if new_cache_config:
        cache = Cache(**new_cache_config)
        set_global_value('cache', cache)
        set_global_value('cache_config', new_cache_config)
    cache_state = CacheState()
    set_global_value('cache_state', cache_state)



def load_cache(namespace:Optional[str]=None) -> Cache|None:
    if namespace:
        cache_path = CacheConst.DEFAULT_CACHE_DIR /namespace
        if cache_path.exists():
            cache_config_path = cache_path / 'config.json'
            with open(cache_config_path) as f:
                cache_config:dict[str,Any] = json.load(f)
            cache = Cache(**cache_config)
            return cache
        else:
            raise ValueError(f'cache with namespace \'{namespace}\' not exist!')
    else:
        cache:Cache = get_global_value('cache')
        return cache


def change_running_cache(namespace:str):
    cache:Cache=load_cache(namespace)#type:ignore
    set_global_value('cache', cache)
    set_global_value('cache_cofig', cache.config)
    _write_last_run_cache_config(cache.config)
    return cache
