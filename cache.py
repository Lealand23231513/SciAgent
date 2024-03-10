from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import logging
from pathlib import Path
import shutil
import json
from langchain.indexes import SQLRecordManager, index
from typing import Any, Literal, cast
from global_var import get_global_value, set_global_value
from config import DEFAULT_CACHE_DIR, EMB_MODEL_MAP, DEFAULT_CACHE_NAMESPACE, DEFAULT_EMB_MODEL_NAME
from typing import cast, List, Optional
import logging


def init_cache(clear_old:bool=False,**kwargs):
    '''
    :param clear_old: if True, will delete old cache if last run cache config is different from given kwargs
    :param kwargs: kwargs of Cache
    '''
    logger = logging.getLogger(Path(__file__).stem+'init_cache')
    default_cache_dir = Path(DEFAULT_CACHE_DIR)
    last_run_cache_config_path = default_cache_dir / "lastrun-settings.json"
    if default_cache_dir.exists() == False:
        default_cache_dir.mkdir(parents=True)
    if last_run_cache_config_path.exists():
        with open(last_run_cache_config_path) as f:
            last_run_cache_config = cast(dict[str, Any], json.load(f))
        if last_run_cache_config!=kwargs:
            if clear_old:
                logger.info(f'last run cache config {last_run_cache_config} is different from given kwargs, so clear last run cache and build a new cache')
                cache=Cache(**last_run_cache_config)
                cache.clear_all()
            last_run_cache_config.update(kwargs)
            with open(last_run_cache_config_path, "w") as f:
                json.dump(last_run_cache_config, f)
        cache = Cache(**last_run_cache_config)
    else:
        logger.info("Can't find last run config, will create a config file")
        last_run_cache_config_path.touch()
        cache = Cache(**kwargs)
        last_run_cache_config = {
            "namespace": cache.namespace,
            "emb_model_name": cache.emb_model_name,
        }
        with open(last_run_cache_config_path, "w") as f:
            json.dump(last_run_cache_config, f)
    set_global_value('cache', cache)
    set_global_value('cache_config', last_run_cache_config)
    return cache



def load_cache():
    try:
        cache = cast(Cache, get_global_value("cache"))
    except Exception as e:
        logger = logging.getLogger(Path(__file__).stem)
        logger.error(repr(e))
        raise e
    return cache


class Cache(object):
    def __init__(
        self,
        namespace: str = DEFAULT_CACHE_NAMESPACE,
        emb_model_name: str = DEFAULT_EMB_MODEL_NAME,
        all_files: Optional[List[str]] = None,
        vectorstore=None,
        record_manager=None,
    ) -> None:
        self.emb_model_name = emb_model_name
        self.namespace = namespace
        self.root_dir = Path(DEFAULT_CACHE_DIR) / namespace / emb_model_name
        if self.root_dir.exists() == False:
            self.root_dir.mkdir(parents=True)
        self.filenames_save_path = self.root_dir / "cached-files.json"
        if self.filenames_save_path.exists() == False:
            self.filenames_save_path.touch()
            with open(self.filenames_save_path,'w') as f:
                json.dump([],f)
        self.cached_files_dir = self.root_dir / "cached-files"
        if self.cached_files_dir.exists() == False:
            self.cached_files_dir.mkdir(parents=True)
        self.embedding = OpenAIEmbeddings(
            model=emb_model_name,
            api_key=EMB_MODEL_MAP[emb_model_name]["api_key"],
            base_url=EMB_MODEL_MAP[emb_model_name]["base_url"],
        )
        # TODO: customed embedding
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
            logger = logging.getLogger(Path(__file__).stem+'.load_filenames')
            logger.info(repr(e))
            self.all_files = []
        return self.all_files

    def load_record_manager(self):
        # load SQL record manager
        self.record_manager = SQLRecordManager(
            self.namespace + self.emb_model_name,
            db_url="sqlite:///"
            + f"{DEFAULT_CACHE_DIR}/{self.namespace}/{self.emb_model_name}/record_manager_cache.sql",
        )
        self.record_manager.create_schema()
        return self.record_manager

    def load_vectorstore(self):
        # load Chroma vecotrstore with OpenAI embeddings
        persist_directory = str(self.root_dir / "chroma")
        self.vectorstore = Chroma(
            collection_name=self.emb_model_name,
            embedding_function=self.embedding,
            persist_directory=persist_directory,
        )
        return self.vectorstore

    def save_filenames(self):
        # save cached files' name
        with open(self.filenames_save_path, "w") as f:
            json.dump(self.all_files, f)

    def clear_all(self):
        # clear all cached data
        # TODO clear cached files
        self.all_files = []
        self.save_filenames()
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
        return res

    def cache_file(
        self,
        path: str,
        save_local=False,
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ):
        """
        :param path: path or url of the file
        :param save_local: whether or not save the uploaded file in local cache folder
        :param chunk_size: max length of the chunk
        """
        if save_local == True:
            shutil.copy(path, Path(DEFAULT_CACHE_DIR + "/cached-files"))
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
        for i in range(len(raw_docs)):
            raw_docs[i].metadata["source"] = Path(raw_docs[i].metadata["source"]).name
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        self.save_filenames()
        logger = logging.getLogger(Path(__file__).stem)
        logger.debug(f"all files:{self.all_files}")
        logger.info(index_res)
        logger.info(f"The file ({Path(path).name}) has been cached")
        return index_res

    def delete_file(self, filename: str):
        logger = logging.getLogger(Path(__file__).stem)
        uids_to_delete = self.record_manager.list_keys(group_ids=[filename])
        num_deleted = 0
        if uids_to_delete:
            self.vectorstore.delete(uids_to_delete)
            self.record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)
            all_files_set = set(self.all_files)
            all_files_set.discard(filename)
            self.all_files = list(all_files_set)
            logger.info(f"The file ({filename}) has been deleted")
        delete_res = {
            "num_added": 0,
            "num_updated": 0,
            "num_skipped": 0,
            "num_deleted": num_deleted,
        }
        logger.info(delete_res)
        return delete_res

    # def __deepcopy__(self, memo=None):
    #     '''
    #     Out of use
    #     '''
    #     from copy import deepcopy
    #     newCache = Cache(
    #         all_files=deepcopy(self.all_files)
    #         )
    #     return newCache
