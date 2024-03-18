import gradio as gr
import logging
from controler import call_agent
from pathlib import Path
from dotenv import load_dotenv
import os
from config import *
from gradio_mchatbot import MultiModalChatbot
from typing import Any, cast, Generator
import global_var
from cache import init_cache, load_cache
from channel import load_channel
import json
import time
from cache import Cache
import wenet
import state

from gradio_mypdf import PDF
from gradio_modal import Modal
from document_qa import document_qa_fn
import numpy as np
from PIL import Image
from multimodal import multimodal_chat
from llm_state import LLMState, LLMConst
from tools import RetrievalState, ToolsState, WebSearchState, RetrievalStateConst, WebSearchStateConst


def _init_state_vars():
    global _state, _timestamp
    _state = {"type": None, "name": None, "message": None}
    _timestamp = str(time.time())
    global_var.set_global_value("state_mutex", False)
    global_var.set_global_value("llm_state", LLMState())
    global_var.set_global_value("retrieval_state", RetrievalState())
    global_var.set_global_value("websearch_state", WebSearchState())
    global_var.set_global_value("tools_state", ToolsState())
    global_var.set_global_value("chat_history", [])


def vqa_chat_submit(
    history: list, user_input: str, imgfile, model: str, api_key: str, base_url: str
):
    user_message = [user_input, {"type": "file", "filepath": imgfile.name}]

    history.append([user_message, None])
    yield history, gr.Textbox(interactive=False)
    stream_response = multimodal_chat(
        user_message, model=model, stream=True, api_key=api_key, base_url=base_url
    )
    stream_response = cast(Generator, stream_response)
    response_message = ""
    for delta in stream_response:
        response_message += delta
        history[-1][1] = response_message
        yield history, None
    logger.info("vqa chat end")
    logger.info(history)
    yield history, gr.Textbox(interactive=True)


def check_and_clear_pdfqa_history(filepath: str | None, txt: str, chat_history: list):
    if filepath is None:
        return [], None
    return chat_history, txt


def chat_with_document(filepath: str, question: str, chat_history: list):
    chat_history.append([question, None])
    yield chat_history, gr.Textbox(interactive=False)
    answer = document_qa_fn(filepath, question)
    chat_history[-1][1] = ""
    for chunk in answer:
        chat_history[-1][1] += chunk
        yield chat_history, None
    yield chat_history, gr.Textbox(interactive=True)


def delete_all_files():
    cache = load_cache()
    cache.clear_all()
    gr.Info(f"所有缓存文件已被清除。")
    return (
        gr.Dropdown(
            label="选择要从本地数据库中删除的文章",
            choices=cache.all_files,  # type:ignore
            interactive=True,
        ),
        [],
    )


def delete_files(filenames: list[str]):
    cache = load_cache()
    for filename in filenames:
        cache.delete_file(filename)
        gr.Info(f"文件 《{filename}》已经被删除")
    logger.info(cache.all_files)
    return gr.Dropdown(
        label="选择要从本地数据库中删除的文章",
        choices=cache.all_files,  # type:ignore
        interactive=True,
    ), [[i] for i in cache.all_files]


def upload(files: list[str]):
    cache = load_cache()
    logger.info(files)
    for filepath in files:
        cache.cache_file(filepath, save_local=True)
        gr.Info(f"文件 {Path(filepath).name} 上传成功!")

    return gr.Dropdown(
        label="选择要从本地数据库中删除的文章",
        choices=cache.all_files,  # type:ignore
        interactive=True,
    ), [[i] for i in cache.all_files]


def change_cache_config(emb_model: str, namespace: str):
    cache = init_cache(namespace=namespace, emb_model_name=emb_model)
    gr.Info("成功更改缓存设置")
    return [[i] for i in cache.all_files]


def wav2txt(path: str, lang: str = "chinese"):
    model = wenet.load_model(lang)
    result = model.transcribe(path)
    return result["text"]


def submit(chatbot, user_input, user_input_wav):
    state.StateMutex.set_state_mutex(True)
    chat_history = cast(list, global_var.get_global_value("chat_history"))
    chat_history.append({"role": "user", "content": f"{user_input}"})
    if user_input_wav:
        user_input = wav2txt(user_input_wav)
    chatbot.append((user_input, None))
    yield chatbot,None,None
    full_response = ""
    for ai_response in call_agent(
        user_input,
        stream=True,
    ):
        full_response += str(ai_response)
        chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot,None,None
    chat_history.append({"role": "assistant", "content": f"{full_response}"})
    logger.info("submit end")
    state.StateMutex.set_state_mutex(False)


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
        return {modal: Modal(visible=False), modalMsg: ""}

    with gr.Blocks(title="SciAgent", theme="soft") as demo:
        with gr.Tab(label="问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SciAgent")
                    with gr.Row():
                        txtbot = gr.Textbox(
                            label="用户对话框:",
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
                        llm_state = cast(
                            LLMState, global_var.get_global_value("llm_state")
                        )
                        llmDdl = gr.Dropdown(
                            choices=SUPPORT_LLMS,  # type: ignore
                            value=llm_state.model,
                            label="大语言模型ID",
                        )
                        llmApikeyDdl = gr.Textbox(
                            label="模型api-key",
                            value=LLMConst.DEFAULT_API_KEY,
                            type="password",
                        )
                        llmBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            value=LLMConst.DEFAULT_BASE_URL,
                            info="如使用Openai模型此栏请留空",
                        )
                        gr.on(
                            [
                                llmDdl.change,
                                llmApikeyDdl.change,
                                llmBaseurlTxt.change,
                            ],
                            lambda model, api_key, base_url: state.change_state(
                                "llm_state",
                                model=model,
                                api_key=api_key,
                                base_url=base_url,
                            ),
                            inputs=[llmDdl, llmApikeyDdl, llmBaseurlTxt],
                        )
                    with gr.Accordion(label="缓存设置"):
                        cache_config = cast(
                            dict[str, Any], global_var.get_global_value("cache_config")
                        )
                        namespaceTxt = gr.Textbox(
                            value=cast(str, cache_config["namespace"]),
                            label="数据库名称",
                        )
                        embDdl = gr.Dropdown(
                            choices=cast(
                                list[str | int | float | tuple[str, str | int | float]]
                                | None,
                                SUPPORT_EMBS,
                            ),
                            value=cache_config["emb_model_name"],
                            label="Embedding模型",
                        )
                        changeCacheBtn = gr.Button(value="切换本地数据库设置")
                    with gr.Accordion(label="搜索设置", open=False):
                        downloadChk = gr.Checkbox(
                            label="下载",
                            value=WebSearchStateConst.DEFAULT_DOWNLOAD
                        )
                        downloadChk.change(
                            lambda download: state.change_state('websearch_state', download=download),
                            inputs=[downloadChk]
                        )
                    with gr.Accordion(label="RAG设置", open=False):
                        temperatureSlider = gr.Slider(
                            label="temperature",
                            minimum=RetrievalStateConst.MIN_TEMPERATURE,
                            maximum=RetrievalStateConst.MAX_TEMPERATURE,
                            value=RetrievalStateConst.DEFAULT_TEMPERATURE,
                            info=f"请在0至1之间选择",
                            interactive=True,
                        )
                        toppSlider = gr.Slider(
                            label="top_p",
                            minimum=RetrievalStateConst.MIN_TOP_P,
                            maximum=RetrievalStateConst.MAX_TOP_P,
                            value=RetrievalStateConst.DEFAULT_TOP_P,
                            step=0.01,
                            info=f"请在0至1之间选择",
                            interactive=True,
                        )
                        chunkSizeSlider = gr.Slider(
                            label="切片长度",
                            minimum=RetrievalStateConst.MIN_CHUNK_SIZE,
                            maximum=RetrievalStateConst.MAX_CHUNK_SIZE,
                            step=1,
                            value=RetrievalStateConst.DEFAULT_CHUNK_SIZE,
                            info="选择每段被切割文案的长度",
                            interactive=True,
                        )
                        scoreThresholdSlider = gr.Slider(
                            label="分数阈值",
                            minimum=RetrievalStateConst.MIN_SCORE_THRESHOLD,
                            maximum=RetrievalStateConst.MAX_SCORE_THRESHOLD,
                            value=RetrievalStateConst.DEFAULT_SCORE_THRESHOLD,
                            interactive=True,
                        )
                        chunkOverlapSlider = gr.Slider(
                            label="切片重叠部分长度",
                            minimum=RetrievalStateConst.MIN_CHUNK_OVERLAP,
                            maximum=RetrievalStateConst.MAX_CHUNK_OVERLAP,
                            step=50,
                            value=RetrievalStateConst.DEFAULT_CHUNK_OVERLAP,
                            interactive=True,
                        )
                        gr.on(
                            [
                                temperatureSlider.change,
                                toppSlider.change,
                                chunkSizeSlider.change,
                                scoreThresholdSlider.change,
                                chunkOverlapSlider.change,
                            ],
                            lambda temperature, top_p, chunk_size, score_threshold, chunk_overlap: state.change_state(
                                "retrieval_state",
                                temperature=temperature,
                                top_p=top_p,
                                chunk_size=chunk_size,
                                score_threshold=score_threshold,
                                chunk_overlap=chunk_overlap,
                            ),
                            inputs=[
                                temperatureSlider,
                                toppSlider,
                                chunkSizeSlider,
                                scoreThresholdSlider,
                                chunkOverlapSlider,
                            ],
                        )

        with gr.Tab(label="本地数据库"):
            cache = load_cache()
            with gr.Row():
                with gr.Column(scale=3):
                    dstCachedPapers = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        label="已存储的文章",
                        samples=[[i] for i in cache.all_files],
                    )
                with gr.Column(scale=1):
                    uploadFileBot = gr.File(
                        label="上传文件",
                        file_types=[".pdf", ".docx"],
                        file_count="multiple",
                    )
                    cacheFilesDdl = gr.Dropdown(
                        label="选择要从本地数据库中删除的文章",
                        choices=cache.all_files,  # type:ignore
                        multiselect=True,
                    )
                    delete_button = gr.Button("删除")
                    cleanCacheBtn = gr.Button("删除全部")
            delete_button.click(
                fn=delete_files,
                inputs=[cacheFilesDdl],
                outputs=[cacheFilesDdl, dstCachedPapers],
            )
            cleanCacheBtn.click(
                delete_all_files, outputs=[cacheFilesDdl, dstCachedPapers]
            )
            uploadFileBot.upload(
                upload, inputs=[uploadFileBot], outputs=[cacheFilesDdl, dstCachedPapers]
            ).then(lambda: gr.update(value=None), outputs=[uploadFileBot])
        with gr.Tab(label="工作台"):
            pass
        with gr.Tab(label="PDF文档问答"):
            with gr.Row():
                with gr.Column():
                    pdfBox = PDF(label="PDF文档", height=700)
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
                    multiModalChatbot = MultiModalChatbot(
                        label="SciAgent-V"
                    )
                    gr.Markdown(
                        "输入问题并上传图片后，点击提交开始视觉问答。注意：目前暂不支持多轮对话。"
                    )
                    with gr.Column(scale=3):
                        vqaTxtbot = gr.Textbox(
                            label="用户对话框:", placeholder="在这里输入", lines=4
                        )
                    with gr.Column(scale=1):
                        vqaImgBox = gr.File(label="图片", file_types=["image"])
                    with gr.Row():
                        vqaClearBtn = gr.ClearButton(
                            value="清除对话记录",
                            components=[multiModalChatbot, vqaTxtbot, vqaImgBox],
                        )
                        vqaSubmitBtn = gr.Button("提交")
                with gr.Column(scale=1):
                    with gr.Accordion(label="模型设置"):
                        mllmDdl = gr.Dropdown(
                            choices=cast(
                                list[str | int | float | tuple[str, str | int | float]]
                                | None,
                                SUPPORT_MLLMS,
                            ),
                            value=SUPPORT_MLLMS[0],
                            label="多模态大模型ID",
                        )
                        mllmApikeyDdl = gr.Textbox(label="模型api-key", type="password")
                        mllmBaseurlTxt = gr.Textbox(
                            label="模型baseurl", info="如使用Openai模型此栏请留空"
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

        changeCacheBtn.click(
            change_cache_config,
            inputs=[embDdl, namespaceTxt],
            outputs=[dstCachedPapers],
        )
        submitBtn.click(
            fn=submit,
            inputs=[chatbot, txtbot, audio],
            outputs=[chatbot, txtbot, audio]
        )
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
        vqaSubmitBtn.click(
            vqa_chat_submit,
            inputs=[
                multiModalChatbot,
                vqaTxtbot,
                vqaImgBox,
                mllmDdl,
                mllmApikeyDdl,
                mllmBaseurlTxt,
            ],
            outputs=[multiModalChatbot, vqaTxtbot],
        )
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
