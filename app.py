import gradio as gr
import logging

from controler import call_agent, AGENT_DONE
from pathlib import Path
from dotenv import load_dotenv
import os
from config import *
from gradio_mchatbot import MultiModalChatbot
from typing import Any, Optional, cast, Generator
import global_var
from cache import (
    init_cache,
    load_cache,
    Cache,
    CacheConst,
    change_running_cache,
)
from channel import load_channel
import json
import time
from cache import Cache
import wenet
import state

from gradio_mypdf import PDF
from gradio_modal import Modal
from document_qa import document_qa_fn
from multimodal import multimodal_chat
from model_state import (
    LLMState,
    LLMStateConst,
    MLLMState,
    MLLMStateConst,
    EMBStateConst,
    EMBState,
)
from tools import (
    ToolsState,
)
from retrieval_qa import RetrievalState, RetrievalConst

from websearch.const import WebSearchStateConst
from websearch.websearch_state import WebSearchState


def _init_state_vars():
    global _state, _timestamp
    _state = {"type": None, "name": None, "message": None}
    _timestamp = str(time.time())
    global_var.set_global_value("state_mutex", False)
    global_var.set_global_value("llm_state", LLMState())
    global_var.set_global_value("mllm_state", MLLMState())
    global_var.set_global_value("retrieval_state", RetrievalState())
    global_var.set_global_value("websearch_state", WebSearchState())
    global_var.set_global_value("tools_state", ToolsState())
    global_var.set_global_value("chat_history", [])
    global_var.set_global_value("pdf_llm_state", LLMState())
    global_var.set_global_value("new_emb_state", EMBState())


def wav2txt(path: str, lang: str = "chinese"):
    model = wenet.load_model(lang)
    result = model.transcribe(path)
    return result["text"]


def submit(chatbot, user_input, user_input_wav, exe_log: str | None):
    state.StateMutex.set_state_mutex(True)
    chat_history = cast(list, global_var.get_global_value("chat_history"))
    chat_history.append({"role": "user", "content": f"{user_input}"})
    if user_input_wav:
        user_input = wav2txt(user_input_wav)
    if isinstance(exe_log, str) == False:
        exe_log = ""
    chatbot.append((user_input, None))
    yield chatbot, None, None, exe_log
    exe_log = cast(str, exe_log)
    generator = call_agent(
        user_input,
        stream=True,
    )
    full_response = ""
    for ai_response in generator:
        ai_response = cast(str, ai_response)
        exe_log += ai_response + "\n"
        if ai_response == AGENT_DONE:
            full_response = next(generator)
            chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot, None, None, exe_log
    chat_history.append({"role": "assistant", "content": f"{full_response}"})
    logger.info("submit end")
    state.StateMutex.set_state_mutex(False)


def vqa_chat(history: list, user_input: Optional[str], img_path: Optional[str]):
    state.StateMutex.set_state_mutex(True)
    mllm_state: MLLMState = global_var.get_global_value("mllm_state")
    if img_path:
        user_message = [user_input, {"type": "file", "filepath": img_path}]
    else:
        user_message = [user_input]
    history.append([user_message, None])
    yield history, gr.update(interactive=False)
    stream_response = multimodal_chat(user_message, stream=True)
    stream_response = cast(Generator, stream_response)
    response_message = ""
    for delta in stream_response:
        response_message += delta
        history[-1][1] = response_message
        yield history, None
    logger.info("vqa chat end")
    yield history, gr.update(interactive=True)
    state.StateMutex.set_state_mutex(False)


def check_and_clear_pdfqa_history(filepath: str | None, txt: str, chat_history: list):
    if filepath is None:
        return [], None
    return chat_history, txt


def chat_with_document(filepath: str, question: str, chat_history: list):
    state.StateMutex.set_state_mutex(True)
    chat_history.append([question, None])
    yield chat_history, gr.Textbox(interactive=False)
    answer = document_qa_fn(filepath, question)
    chat_history[-1][1] = ""
    for chunk in answer:
        chat_history[-1][1] += chunk
        yield chat_history, None
    yield chat_history, gr.Textbox(interactive=True)
    state.StateMutex.set_state_mutex(False)


def on_select(evt: gr.SelectData):
    cache: Cache = global_var.get_global_value("cache")
    gr.Info(f"已在PDF阅读器中打开“{evt.value[0]}”，切换到“PDF文档问答”栏以查看详情")
    return str(cache.cached_files_dir / evt.value[0])


def delete_all_files():
    cache = load_cache()
    if cache is None:
        raise gr.Error("请先建立知识库")
    cache.clear_all()
    gr.Info(f"所有文章都已被删除")
    return (
        gr.Dropdown(
            label="选择要从本地知识库中删除的文章",
            choices=cache.all_files,  # type:ignore
            interactive=True,
        ),
        [],
    )


def delete_files(filenames: list[str]):
    cache = load_cache()
    if cache is None:
        raise gr.Error("请先建立知识库")
    if isinstance(filenames, list) == False:
        gr.Info("请选择要删除的文章")
        filenames = []
    for filename in filenames:
        cache.delete_file(filename)
        gr.Info(f"文件 ({filename})已经被删除")
    logger.debug(cache.all_files)
    return gr.update(
        choices=cache.all_files,  # type:ignore
    ), [[i] for i in cache.all_files]


def upload(files: list[str]):
    cache = load_cache()
    if cache is None:
        raise gr.Error("请先建立知识库！")
    logger.info(f"upload files:{files}")
    for filepath in files:
        cache.cache_file(filepath, save_local=True)
        gr.Info(f"文件 {Path(filepath).name} 上传成功!")

    return gr.update(
        choices=cache.all_files,  # type:ignore
    ), [[i] for i in cache.all_files]


def create_cache(namespace: str):
    state.StateMutex.set_state_mutex(True)
    valid = True
    new_emb_state: EMBState = global_var.get_global_value("new_emb_state")
    if isinstance(namespace, str) == False or namespace == "":
        gr.Info("请输入知识库的名字")
        valid = False
    if isinstance(new_emb_state.model, str) == False or new_emb_state.model == "":
        gr.Info("请选择Embbeding模型")
        valid = False
    if isinstance(new_emb_state.api_key, str) == False or new_emb_state.api_key == "":
        gr.Info("请输入api-key模型")
        valid = False
    if valid == False:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    if isinstance(new_emb_state.base_url, str) == False or new_emb_state.base_url == "":
        new_emb_state.base_url = None
    cache_lst: list[str] = global_var.get_global_value("cache_lst")
    if namespace in cache_lst:
        gr.Warning(f"知识库“{namespace}”已经存在，故无法创建")
        cache = load_cache(namespace)
    else:
        cache = Cache(
            namespace=namespace,
            emb_model_name=new_emb_state.model,
            emb_api_key=new_emb_state.api_key,
            emb_base_url=new_emb_state.base_url,
        )
        gr.Info(f"新的知识库“{namespace}”创建成功")
    state.StateMutex.set_state_mutex(False)
    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None, choices=cache_lst),
        gr.update(value=None),
        gr.update(value=None),
    )


def change_cache(namespace: str):
    if isinstance(namespace, str) == False:
        raise gr.Error("请选择知识库！")
    cache = change_running_cache(namespace)
    gr.Info("成功更改本地知识库设置")
    return (
        gr.update(value=cache.namespace),
        gr.update(value=cache.emb_model_name),
        gr.update(value=None, choices=cache.all_files),
        [[i] for i in cache.all_files],
    )


def delete_cache(namespace: str):
    cache: Cache = global_var.get_global_value("cache")
    if cache is None:
        raise gr.Error("请先建立知识库！")
    if cache.namespace == namespace:
        gr.Info("无法删除正在使用的本地知识库")
        return gr.update()
    elif isinstance(namespace, str):
        cache = cast(Cache, load_cache(namespace=namespace))
        cache.delete_cache()
        cache_lst: list[str] = global_var.get_global_value("cache_lst")
        gr.Info(f"成功删除本地知识库{namespace}")
        return gr.update(value=None, choices=cache_lst)
    else:
        raise gr.Error("请选择知识库！")


def confirmBtn_click():
    global _state
    response_msg = json.dumps({**_state, "response": True})
    channel = load_channel()
    channel.send(response_msg, "front")
    return Modal(visible=False)


def modal_blur():
    global _state
    response_msg = json.dumps({**_state, "response": False})
    channel = load_channel()
    channel.send(response_msg, "front")
    return Modal(visible=False)


def get_timestamp():
    global _timestamp, _state
    channel = load_channel()
    response = channel.recv("front")
    if response is not None:
        _timestamp = str(time.time())
        _state = json.loads(response)
    else:
        _state = {"type": None, "name": None, "message": None}
    return _timestamp


def create_ui():
    def _state_change():
        global _state
        if _state["type"] == "funcall":
            if _state["name"] == "confirm":
                return {
                    modal: Modal(visible=True, allow_user_close=False),
                    modalMsg: _state["message"],
                }
        if _state["type"] == "modal":
            if _state["name"] == "error":
                raise gr.Error(cast(str, _state["message"]))
            if _state["name"] == "info":
                gr.Info(cast(str, _state["message"]))
            if _state["name"] == "warning":
                gr.Warning(cast(str, _state["message"]))
        return {modal: Modal(visible=False), modalMsg: ""}

    def load_demo():
        cache: Cache | None = global_var.get_global_value("cache")
        cache_lst: list[str] = global_var.get_global_value("cache_lst")
        mllm_state: MLLMState = global_var.get_global_value("mllm_state")
        llm_state: LLMState = global_var.get_global_value("llm_state")
        pdf_llm_state: LLMState = global_var.get_global_value("pdf_llm_state")
        return {
            dstCachedPapers: [[i] for i in cache.all_files] if cache else [],
            cacheFilesDdl: (
                gr.update(value=None, choices=cache.all_files) if cache else []
            ),
            cacheNameDdl: (
                gr.update(value=cache.namespace, choices=cache_lst) if cache else None
            ),
            currEmbTxt: cache.emb_model_name if cache else None,
            currNamespaceTxt: cache.namespace if cache else None,
            mllmDdl: mllm_state.model,
            mllmApikeyDdl: mllm_state.api_key,
            mllmBaseurlTxt: mllm_state.base_url,
            llmDdl: llm_state.model,
            llmApikeyDdl: llm_state.api_key,
            llmBaseurlTxt: llm_state.base_url,
            pdfLlmDdl: pdf_llm_state.model,
            pdfLlmApikeyDdl: pdf_llm_state.api_key,
            pdfLlmBaseurlTxt: pdf_llm_state.base_url,
        }

    with gr.Blocks(title="SciAgent", theme="soft") as demo:
        with gr.Tab(label="使用Agent"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SciAgent")
                    with gr.Row():
                        txtbot = gr.Textbox(
                            label="用户对话框",
                            placeholder="在这里输入",
                            lines=8,
                            scale=3,
                        )
                        audio = gr.Audio(
                            label="语音输入", type="filepath", format="wav", scale=1
                        )
                    with gr.Row():
                        clearBtn = gr.ClearButton(
                            value="清除对话记录",
                            components=[chatbot],
                        )
                        submitBtn = gr.Button("提交")
                        clearBtn.click(
                            fn=lambda: global_var.set_global_value("chat_history", [])
                        )
                    exeLogTxt = gr.Textbox(
                        label="Agent执行记录",
                        value="\n",
                        interactive=False
                    )
                with gr.Column(scale=1):
                    tools_state = cast(
                        ToolsState, global_var.get_global_value("tools_state")
                    )
                    toolsDdl = gr.Dropdown(
                        choices=tools_state.tools_choices,  # type:ignore
                        value=tools_state.tools_select,  # type:ignore
                        multiselect=True,
                        label="工具",
                        info="选择要使用的工具",
                        interactive=True,
                    )
                    toolsDdl.change(
                        lambda tools_select: state.change_state(
                            "tools_state", tools_select=tools_select
                        ),
                        inputs=[toolsDdl],
                    )
                    with gr.Accordion(label="模型设置"):
                        llmDdl = gr.Dropdown(
                            choices=LLMStateConst.LLM_CHOICES,  # type: ignore
                            # value=llm_state.model,
                            label="模型ID",
                        )
                        llmApikeyDdl = gr.Textbox(
                            label="模型api-key",
                            value=LLMStateConst.DEFAULT_API_KEY,
                            type="password",
                        )
                        llmBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            value=LLMStateConst.DEFAULT_BASE_URL,
                            info="如使用Openai模型此栏请留空",
                        )  # TODO temperature, top_p control
                        llmTempSlider = gr.Slider(
                            label="temperature",
                            minimum=LLMStateConst.MIN_TEMPERATURE,
                            maximum=LLMStateConst.MAX_TEMPERATURE,
                            value=LLMStateConst.DEFAULT_TEMPERATURE,
                            info=f"请在0至1之间选择",
                            interactive=True,
                        )
                        gr.on(
                            [
                                llmDdl.change,
                                llmApikeyDdl.change,
                                llmBaseurlTxt.change,
                                llmTempSlider.change,
                            ],
                            lambda model, api_key, base_url, temperature: state.change_state(
                                "llm_state",
                                model=model,
                                api_key=api_key,
                                base_url=base_url,
                                temperature=temperature,
                            ),
                            inputs=[llmDdl, llmApikeyDdl, llmBaseurlTxt, llmTempSlider],
                        )
                    with gr.Accordion(label="搜索设置", open=False):
                        with gr.Row():
                            downloadChk = gr.Checkbox(
                                label="下载并存入知识库",
                                value=WebSearchStateConst.DEFAULT_DOWNLOAD,
                            )
                            enableSeSearchChk = gr.Checkbox(
                                label="开启联网搜索",
                                info="选择后Agent可以选择在搜索引擎搜索信息",
                                value=WebSearchStateConst.DEFAULT_ENABLE_SE_SEARCH,
                            )
                        with gr.Row():
                            paperSearchPlatformDdl = gr.Dropdown(
                                label="论文检索平台",
                                scale=1,
                                value=WebSearchStateConst.DEFAULT_PAPER_SEARCH_SELECT,
                                choices=WebSearchStateConst.PAPER_SEARCH_CHOICES,  # type:ignore
                            )
                            seDdl = gr.Dropdown(
                                label="搜索引擎",
                                scale=1,
                                value=WebSearchStateConst.DEFAULT_SE_SELECT,
                                choices=WebSearchStateConst.SE_CHOICES,  # type:ignore
                            )
                        gr.on(
                            [
                                downloadChk.change,
                                paperSearchPlatformDdl.change,
                                enableSeSearchChk.change,
                                seDdl.change,
                            ],
                            lambda download, paper_search_tool_select, enable_se_search, se_search_tool_select: state.change_state(
                                "websearch_state",
                                download=download,
                                paper_search_tool_select=paper_search_tool_select,
                                enable_se_search=enable_se_search,
                                se_search_tool_select=se_search_tool_select,
                            ),
                            inputs=[
                                downloadChk,
                                paperSearchPlatformDdl,
                                enableSeSearchChk,
                                seDdl,
                            ],
                        )
                    with gr.Accordion(label="RAG设置", open=False):
                        strategyDdl = gr.Dropdown(
                            label="retrieval策略",
                            value=RetrievalConst.DEFAULT_STRATEGY,
                            choices=RetrievalConst.STRATEGIES_CHOICES,  # type:ignore
                            interactive=True,
                        )
                        kNum = gr.Number(
                            label="文段数量",
                            minimum=RetrievalConst.MINIMUM_K,
                            maximum=RetrievalConst.MAXIMUN_K,
                            value=RetrievalConst.DEFAULT_K,
                            info=f"retriever从本地知识库中抽取的文段数量",
                            interactive=True,
                            precision=0
                        )
                        scoreThresholdSlider = gr.Slider(
                            label="分数阈值",
                            minimum=RetrievalConst.MIN_SCORE_THRESHOLD,
                            maximum=RetrievalConst.MAX_SCORE_THRESHOLD,
                            value=RetrievalConst.DEFAULT_SCORE_THRESHOLD,
                            interactive=True,
                        )

                        gr.on(
                            [
                                scoreThresholdSlider.change,
                                strategyDdl.change,
                                kNum.change
                            ],
                            lambda score_threshold, strategy, k: state.change_state(
                                "retrieval_state",
                                score_threshold=score_threshold,
                                strategy=strategy,
                                k=k
                            ),
                            inputs=[
                                scoreThresholdSlider,
                                strategyDdl,
                                kNum
                            ],
                        )

        with gr.Tab(label="本地知识库"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("可以点击文章名查看文章内容")
                    dstCachedPapers = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        label="已存储的文章",
                        # samples=[[i] for i in cache.all_files],
                        samples=[],
                    )
                with gr.Column(scale=1):
                    with gr.Accordion(label="新建知识库", open=False):
                        newNamespaceTxt = gr.Textbox(
                            label="知识库名称",
                            info="名称应为3-63个字母或者数字组成的字符串",
                        )
                        newEmbDdl = gr.Dropdown(
                            choices=EMBStateConst.EMB_CHOICES,  # type:ignore
                            multiselect=False,
                            label="Embedding模型",
                        )
                        newEmbApiKeyDdl = gr.Textbox(
                            label="模型api-key",
                            value=EMBStateConst.DEFAULT_API_KEY,
                            type="password",
                        )
                        newEmbBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            value=EMBStateConst.DEFAULT_BASE_URL,
                            info="如使用Openai模型此栏请留空",
                        )
                        newCacheButton = gr.Button(value="新建本地知识库")
                    currNamespaceTxt = gr.Textbox(
                        label="当前知识库",
                        # value=cache.namespace,
                        interactive=False,
                    )
                    currEmbTxt = gr.Textbox(
                        label="当前知识库使用的Embedding模型",
                        # value=cache.emb_model_name,
                        interactive=False,
                    )
                    gr.on(
                            [
                                newEmbDdl.change,
                                newEmbApiKeyDdl.change,
                                newEmbBaseurlTxt.change,
                            ],
                            lambda model, api_key, base_url: state.change_state(
                                "new_emb_state",
                                model=model,
                                api_key=api_key,
                                base_url=base_url,
                            ),
                            inputs=[newEmbDdl, newEmbApiKeyDdl, newEmbBaseurlTxt],
                        )
                    with gr.Accordion(label="文件管理"):
                        chunkSizeSlider = gr.Slider(
                            label="切片长度",
                            minimum=CacheConst.MIN_CHUNK_SIZE,
                            maximum=CacheConst.MAX_CHUNK_SIZE,
                            step=1,
                            value=CacheConst.DEFAULT_CHUNK_SIZE,
                            info="选择每段被切割文案的长度",
                            interactive=True,
                        )
                        chunkOverlapSlider = gr.Slider(
                            label="切片重叠部分长度",
                            minimum=CacheConst.MIN_CHUNK_OVERLAP,
                            maximum=CacheConst.MAX_CHUNK_OVERLAP,
                            step=50,
                            value=CacheConst.DEFAULT_CHUNK_OVERLAP,
                            interactive=True,
                        )
                        cacheFilesDdl = gr.Dropdown(
                            label="选择要从本地知识库中删除的文章",
                            multiselect=True,
                        )
                        uploadFileBot = gr.File(
                            label="上传文件",
                            file_types=[".pdf", ".docx"],
                            file_count="multiple",
                        )
                        delete_button = gr.Button("删除")
                        cleanCacheBtn = gr.Button("删除全部")
                        gr.on(
                            [chunkSizeSlider.change, chunkOverlapSlider.change],
                            fn=lambda chunk_size, chunk_overlap: state.change_state(
                                "cache_state",
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                            ),
                            inputs=[chunkSizeSlider, chunkOverlapSlider],
                        )
                        delete_button.click(
                            fn=delete_files,
                            inputs=[cacheFilesDdl],
                            outputs=[cacheFilesDdl, dstCachedPapers],
                        )
                        cleanCacheBtn.click(
                            delete_all_files, outputs=[cacheFilesDdl, dstCachedPapers]
                        )
                        uploadFileBot.upload(
                            upload,
                            inputs=[uploadFileBot],
                            outputs=[cacheFilesDdl, dstCachedPapers],
                        ).then(lambda: gr.update(value=None), outputs=[uploadFileBot])

                    with gr.Accordion(label="更改知识库", open=False):
                        cacheNameDdl = gr.Dropdown(
                            # value=cache.namespace,
                            # choices=cache_lst,  # type:ignore
                            label="选择知识库",
                        )
                        changeCacheBtn = gr.Button(value="更改为选中的本地知识库")
                        deleteCacheBtn = gr.Button(value="删除选中的本地知识库")
                    newCacheButton.click(
                        create_cache,
                        inputs=[newNamespaceTxt],
                        outputs=[
                            newNamespaceTxt,
                            newEmbDdl,
                            cacheNameDdl,
                            newEmbApiKeyDdl,
                            newEmbBaseurlTxt,
                        ],
                    )
                    changeCacheBtn.click(
                        change_cache,
                        inputs=[cacheNameDdl],
                        outputs=[
                            currNamespaceTxt,
                            currEmbTxt,
                            cacheFilesDdl,
                            dstCachedPapers,
                        ],
                    )
                    deleteCacheBtn.click(
                        delete_cache, inputs=[cacheNameDdl], outputs=[cacheNameDdl]
                    )
                    submitBtn.click(
                        fn=submit,
                        inputs=[chatbot, txtbot, audio, exeLogTxt],
                        outputs=[chatbot, txtbot, audio, exeLogTxt],
                        scroll_to_output=True,
                    )

        # with gr.Tab(label="工作台"):
        #     pass
        with gr.Tab(label="PDF文档问答"):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("模型设置"):
                        pdfLlmDdl = gr.Dropdown(
                            choices=LLMStateConst.LLM_CHOICES,  # type:ignore
                            value=LLMStateConst.DEFAULT_LLM,
                            label="模型ID",
                        )
                        pdfLlmApikeyDdl = gr.Textbox(
                            label="模型api-key",
                            type="password",
                            value=LLMStateConst.DEFAULT_API_KEY,
                        )

                        pdfLlmBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            info="如使用Openai模型此栏请留空",  # BUG info不显示
                            value=LLMStateConst.DEFAULT_BASE_URL,
                        )
                        pdfLlmTempSlider = gr.Slider(
                            label="temperature",
                            minimum=LLMStateConst.MIN_TEMPERATURE,
                            maximum=LLMStateConst.MAX_TEMPERATURE,
                            value=LLMStateConst.DEFAULT_TEMPERATURE,
                            info=f"请在0至1之间选择",
                            interactive=True,
                        )
                    gr.Markdown("模型上下文长度有限，请不要传入太长的文章。")
                    pdfBox = PDF(
                        label="PDF文档",
                        height=700,
                        info="模型上下文长度有限，请不要传入太长的文章。",
                    )
                    gr.on(
                        [
                            pdfLlmDdl.change,
                            pdfLlmApikeyDdl.change,
                            pdfLlmBaseurlTxt.change,
                            pdfLlmTempSlider.change,
                        ],
                        lambda model, api_key, base_url, temperature: state.change_state(
                            "pdf_llm_state",
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            temperature=temperature,
                        ),
                        inputs=[
                            pdfLlmDdl,
                            pdfLlmApikeyDdl,
                            pdfLlmBaseurlTxt,
                            pdfLlmTempSlider,
                        ],
                    )
                with gr.Column():
                    docChatbot = gr.Chatbot(label="问答记录", height=800)
                    docTxtbot = gr.Textbox(
                        label="用户对话框:", placeholder="在这里输入", lines=4
                    )
                    # docChatHistory = gr.State([])
                    with gr.Row():
                        docClearBtn = gr.ClearButton(
                            value="清除问答记录", components=[docTxtbot, docChatbot]
                        )
                        docSubmitBtn = gr.Button("提交")
        with gr.Tab(label="视觉问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    multiModalChatbot = MultiModalChatbot(label="SciAgent-V")
                    gr.Markdown(
                        "输入问题并上传图片后，点击提交开始视觉问答。注意：目前暂不支持多轮对话。"
                    )
                    with gr.Column(scale=3):
                        vqaTxtbot = gr.Textbox(
                            label="用户对话框:", placeholder="在这里输入", lines=4
                        )
                    with gr.Column(scale=1):
                        vqaImgBox = gr.Image(label="图片", type="filepath")
                    with gr.Row():
                        vqaClearBtn = gr.ClearButton(
                            value="清除对话记录",
                            components=[multiModalChatbot],
                        )
                        vqaSubmitBtn = gr.Button("提交")
                with gr.Column(scale=1):
                    with gr.Accordion(label="模型设置"):
                        mllmDdl = gr.Dropdown(
                            choices=MLLMStateConst.MLLM_CHOICES,  # type:ignore
                            value=MLLMStateConst.DEFAULT_MLLM,
                            label="多模态大模型ID",
                        )
                        mllmApikeyDdl = gr.Textbox(
                            label="模型api-key",
                            type="password",
                            value=MLLMStateConst.DEFAULT_API_KEY,
                        )
                        mllmBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            info="如使用Openai模型此栏请留空",
                            value=MLLMStateConst.DEFAULT_BASE_URL,
                        )
                        mllmMaxTokens = gr.Slider(
                            label="max tokens",
                            minimum=MLLMStateConst.MINIMUN_MAX_TOKENS,
                            maximum=MLLMStateConst.MAXIMUM_MAX_TOKENS,
                            value=MLLMStateConst.DEFAULT_MAX_TOKENS,
                        )
                        gr.on(
                            [
                                mllmDdl.change,
                                mllmApikeyDdl.change,
                                mllmBaseurlTxt.change,
                                mllmMaxTokens.change,
                                demo.load,
                            ],
                            lambda model, api_key, base_url, max_tokens: state.change_state(
                                "mllm_state",
                                model=model,
                                api_key=api_key,
                                base_url=base_url,
                                max_tokens=max_tokens,
                            ),
                            inputs=[
                                mllmDdl,
                                mllmApikeyDdl,
                                mllmBaseurlTxt,
                                mllmMaxTokens,
                            ],
                        )
                        vqaSubmitBtn.click(
                            vqa_chat,
                            inputs=[
                                multiModalChatbot,
                                vqaTxtbot,
                                vqaImgBox,
                            ],
                            outputs=[multiModalChatbot, vqaTxtbot],
                        )

        # with gr.Column():
        #     state_txt_box = gr.Textbox()

        # this is used for Back-end communication
        timeStampDisp = gr.Textbox(
            label="时间戳", value=get_timestamp, every=1, visible=False
        )
        with Modal(visible=False) as modal:
            modalMsg = gr.Markdown()
            with gr.Row():
                confirmBtn = gr.Button("是")
                cancelBtn = gr.Button("否")
        confirmBtn.click(
            confirmBtn_click, inputs=None, outputs=modal, show_progress="hidden"
        )
        gr.on(
            [cancelBtn.click, modal.blur],
            modal_blur,
            outputs=[modal],  # type: ignore
            show_progress="hidden",
        )
        timeStampDisp.change(_state_change, inputs=None, outputs=[modal, modalMsg])
        gr.on(
            [docTxtbot.submit, docSubmitBtn.click],
            fn=chat_with_document,
            inputs=[pdfBox, docTxtbot, docChatbot],
            outputs=[docChatbot, docTxtbot],
        )
        pdfBox.change(
            check_and_clear_pdfqa_history,
            [pdfBox, docTxtbot, docChatbot],
            [docChatbot, docTxtbot],
        )
        demo.load(
            fn=load_demo,
            outputs=[
                llmDdl,
                llmApikeyDdl,
                llmBaseurlTxt,
                dstCachedPapers,
                cacheFilesDdl,
                cacheNameDdl,
                currEmbTxt,
                currNamespaceTxt,
                pdfLlmDdl,
                pdfLlmApikeyDdl,
                pdfLlmBaseurlTxt,
                mllmDdl,
                mllmApikeyDdl,
                mllmBaseurlTxt,
            ],
        )
        dstCachedPapers.select(on_select, outputs=pdfBox, scroll_to_output=True)
    return demo


# Launch gradio UI
def main():
    load_dotenv()
    global_var._init()
    if Path(DEFAULT_CACHE_DIR).exists() == False:
        Path(DEFAULT_CACHE_DIR).mkdir(parents=True)

    logger.info(f"gradio version: {gr.__version__}")
    load_channel()
    init_cache()
    _init_state_vars()
    demo = create_ui()
    logger.info("SciAgent start!")
    demo.queue().launch(inbrowser=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(Path(__file__).stem)
    main()
