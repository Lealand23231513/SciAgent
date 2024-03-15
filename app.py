import gradio as gr
import logging

from regex import F
from controler import call_agent
from pathlib import Path
from dotenv import load_dotenv
import os
from config import *
from gradio_modal import Modal
from typing import Any, cast, Generator
import global_var
from cache import init_cache, load_cache
from channel import load_channel
import json
import time
from gradio_mypdf import PDF
from document_qa import document_qa_fn
from gradio_mchatbot import MultiModalChatbot
import numpy as np
from PIL import Image
from multimodal import multimodal_chat

def _init_state_vars():
    global state, timestamp
    state = {
        "type": None,
        "name": None,
        "message": None
    }
    timestamp = str(time.time())

def vqa_chat_submit(history:list, user_input:str, imgfile, model:str, api_key:str, base_url:str):
    user_message = [
        user_input,
        {
            "type": "file",
            "filepath": imgfile.name
        }
    ]

    history.append(
        [
            user_message,
            None
        ]
    )
    yield history, gr.Textbox(interactive=False)
    stream_response = multimodal_chat(
        user_message,
        model=model,
        stream=True,
        api_key=api_key,
        base_url=base_url
    )
    stream_response = cast(Generator, stream_response)
    response_message = ''
    for delta in stream_response:
        response_message+=delta
        history[-1][1]=response_message
        yield history, None
    logger.info("vqa chat end")
    logger.info(history)
    yield history, gr.Textbox(interactive=True)


def check_and_clear_pdfqa_history(filepath:str|None, txt:str, chat_history:list):
    if filepath is None:
        return [], None
    return chat_history, txt

def chat_with_document(filepath:str, question:str, chat_history:list):
    chat_history.append([question,None])
    yield chat_history, gr.Textbox(interactive=False)
    answer = document_qa_fn(filepath, question)
    chat_history[-1][1]=''
    for chunk in answer:
        chat_history[-1][1]+=chunk
        yield chat_history, None
    yield chat_history, gr.Textbox(interactive=True)

def clear_cache():
    cache = load_cache()
    cache.clear_all()
    gr.Info(f'所有缓存文件已被清除。')
    return []

def add_input(user_input, chatbot):
    chatbot.append((user_input,None))
    return (                                                                                                                                                                                                        
            gr.Textbox(interactive = False, value = '', placeholder = ''), 
            chatbot, 
            gr.Dropdown(interactive=False),
            gr.Slider(interactive=False)
    )

def change_cache_config(emb_model:str, namespace:str):
    cache = init_cache(namespace=namespace, emb_model_name=emb_model)
    gr.Info("成功更改缓存设置")
    return [[i] for i in cache.all_files]


def submit(
        chatbot, 
        chat_history, 
        toolsDdl:list, 
        downloadChkValue:bool, 
        temperatureValue:float,
        llmDdlValue:str
    ):
    user_input = chatbot[-1][0]
    chat_history.append(
        {
            "role": "user",
            "content": f"{user_input}"
        }
    )
    full_response = ""
    tools = []
    for tool in toolsDdl:
        config = {
            "name":tool,
            "kwargs": {}
        }
        if tool=='websearch':
            config['kwargs']['download']=downloadChkValue
        tools.append(config)
    for ai_response in call_agent(user_input, chat_history, model=llmDdlValue, tools_choice=tools, retrieval_temp=temperatureValue, stream=True):
        full_response += str(ai_response)    #type: ignore
        chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot, chat_history
    chat_history.append(
        {
            "role": "assistant",
            "content": f"{full_response}"
        }
    )
    logger.info("submit end")
    return chatbot, chat_history

def upload(file_obj):
    cache = load_cache()
    cache.cache_file(str(Path(file_obj.name)), save_local=True)
    gr.Info('文件 {} 上传成功!'.format(os.path.basename(Path(file_obj.name))))

    return [[i] for i in cache.all_files]
def confirmBtn_click():
    global state
    response_msg = json.dumps(
        {
            **state,
            "response": True
        }
    )
    channel = load_channel()
    channel.send(response_msg, 'front')
    return Modal(visible=False)
def modal_blur():
    global state
    response_msg = json.dumps(
        {
            **state,
            "response": False
        }
    )
    channel = load_channel()
    channel.send(response_msg, 'front')
    return Modal(visible=False)
def get_timestamp():
    global timestamp, state
    channel = load_channel()
    response = channel.recv('front')
    if response is not None:
        timestamp = str(time.time())
        state = json.loads(response)
    else:
        state = {
            "type": None,
            "name": None,
            "message": None
        }
    return timestamp
    

def create_ui():
    def _state_change():
        global state
        if state['type'] == 'funcall':
            if state['name'] == 'confirm':
                return {
                    modal: Modal(visible=True, allow_user_close=False),
                    modalMsg: state['message']
                }
        return {
            modal: Modal(visible=False),
            modalMsg: ''
        }
    with gr.Blocks(title='SciAgent', theme='soft') as demo:
        with gr.Tab(label='聊天'):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SciAgent", height=900)
                    txtbot = gr.Textbox(label="用户对话框:", placeholder="在这里输入", lines=4)
                    chat_history = gr.State([])
                    with gr.Row():
                        clearBtn = gr.ClearButton(
                            value = "清除对话记录",
                            components = [txtbot,chatbot,chat_history]
                        )
                        submitBtn = gr.Button("提交")
                with gr.Column(scale=1):
                    with gr.Row():
                        zhToolsDdl = gr.Dropdown(
                            cast(list[str | int | float | tuple[str, str | int | float]] | None, TOOLS_LIST),  
                            multiselect=True, 
                            label="工具", 
                            info="选择要使用的工具",
                            interactive=True
                        )
                        
                    with gr.Row():
                        uploadFileBtn = gr.File(
                            label="上传文件", 
                            file_types=['.pdf','.docx']
                        )
                        
                    with gr.Row():
                        cleanCacheBtn = gr.Button('清理所有缓存文件')
                    with gr.Row():
                        llmDdl = gr.Dropdown(
                            choices=cast(list[str | int | float | tuple[str, str | int | float]] | None, SUPPORT_LLMS),
                            value='gpt-3.5-turbo',
                            label="大语言模型"
                        )
                    with gr.Accordion(label='缓存设置'):
                        cache_config = cast(dict[str,Any],global_var.get_global_value('cache_config'))
                        namespaceTxt = gr.Textbox(
                            value=cast(str, cache_config['namespace']),
                            label="数据库名称"
                        )
                        embDdl = gr.Dropdown(
                            choices=cast(list[str | int | float | tuple[str, str | int | float]] | None, SUPPORT_EMBS),
                            value=cache_config['emb_model_name'],
                            label="Embedding模型"
                        )
                        changeCacheBtn = gr.Button(
                            value="切换本地数据库设置"
                        )
                    with gr.Accordion(label='搜索设置', open=False):
                        downloadChk = gr.Checkbox(
                            label='下载',
                        )
                    with gr.Accordion(label='RAG设置', open=False):
                        temperatureSlider = gr.Slider(
                            label="temperature",
                            minimum=0.2,
                            maximum=2.0,
                            value=1,
                            step=0.05,
                            info="请在0.2至2.0之间选择",
                            interactive=True
                        )
                        top_p = gr.Slider(
                            label = "top_p",
                            minimum=0.7,
                            maximum=1.0,
                            value=0.9,
                            step=0.01,
                            info="请在0.7至1.0之间选择",
                            interactive=True
                        )
                        chunk_size = gr.Slider(
                            label="切片长度",
                            minimum=20,
                            maximum=500,
                            step=1,
                            value = 100,
                            info="选择每段被切割文案的长度",
                            interactive=True
                        )
                        score_threshold = gr.Slider(
                            label="分数阈值",
                            minimum=20,
                            maximum=500,
                            step=1,
                            value = 100,
                            interactive=True
                        )
                   
                        
            
                
        with gr.Tab(label='缓存文章'):
            cache = load_cache()
            dstCachedPapers = gr.Dataset(
                components=[gr.Textbox(visible=False)], 
                label='缓存文章',
                samples=[[i] for i in cache.all_files]
            )
        with gr.Tab(label='工作台'):
            pass
        with gr.Tab(label='PDF文档问答'):
            with gr.Row():
                with gr.Column():
                    pdfBox = PDF(label="PDF文档", height=1000)
                with gr.Column():
                    docChatbot = gr.Chatbot(label="问答记录", height=900)
                    docTxtbot = gr.Textbox(label="用户对话框:", placeholder="在这里输入", lines=4)
                    # docChatHistory = gr.State([])
                    with gr.Row():
                        docClearBtn = gr.ClearButton(
                            value = "清除问答记录",
                            components = [docTxtbot,docChatbot]
                        )
                        docSubmitBtn = gr.Button("提交")
        with gr.Tab(label="视觉问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    multiModalChatbot = MultiModalChatbot(label="SciAgent-V", height=900)
                    gr.Markdown("输入问题并上传图片后，点击提交开始视觉问答。注意：目前暂不支持多轮对话。")
                    with gr.Column(scale=3):
                        vqaTxtbot = gr.Textbox(label="用户对话框:", placeholder="在这里输入", lines=4)
                    with gr.Column(scale=1):
                        vqaImgBox = gr.File(label='图片',file_types=["image"])
                    with gr.Row():
                        vqaClearBtn = gr.ClearButton(
                            value = "清除对话记录",
                            components = [multiModalChatbot,vqaTxtbot]
                        )
                        vqaSubmitBtn = gr.Button("提交")
                with gr.Column(scale=1):
                    with gr.Accordion(label="模型设置"):
                        mllmDdl = gr.Dropdown(
                            choices=cast(list[str | int | float | tuple[str, str | int | float]] | None, SUPPORT_MLLMS),
                            value=SUPPORT_MLLMS[0],
                            label="多模态大模型ID"
                        )
                        mllmApikeyDdl = gr.Textbox(
                            label="模型api-key"
                        )
                        mllmBaseurlTxt = gr.Textbox(
                            label="模型baseurl",
                            info="如使用Openai模型此栏请留空"
                        )

        # with gr.Column():
        #     state_txt_box = gr.Textbox()
        timeStampDisp = gr.Textbox(label='时间戳', value=get_timestamp, every=1, visible=False)
        with Modal(visible=False) as modal:
            modalMsg = gr.Markdown()
            with gr.Row():
                confirmBtn = gr.Button("是")
                cancelBtn = gr.Button("否")
        confirmBtn.click(
            confirmBtn_click,
            inputs=None, 
            outputs=modal,
            show_progress='hidden'
        )
        gr.on(
            [cancelBtn.click, modal.blur],
            modal_blur,
            outputs=[modal],# type: ignore
            show_progress='hidden'
        )
        timeStampDisp.change(
            _state_change,
            inputs=None,
            outputs=[modal, modalMsg]
        )
        uploadFileBtn.upload(
            upload,
            inputs=[uploadFileBtn], 
            outputs=[dstCachedPapers],
            queue=True
        )
        changeCacheBtn.click(
            change_cache_config,
            inputs=[embDdl,namespaceTxt],
            outputs=[dstCachedPapers]
        )
        cleanCacheBtn.click(
            clear_cache,
            inputs=[],
            outputs=[dstCachedPapers]
        )

        submitBtn.click(
            fn = add_input, 
            inputs = [txtbot, chatbot], 
            outputs = [txtbot, chatbot, zhToolsDdl, temperatureSlider]
        ).then(
            fn = submit, 
            inputs = [chatbot, chat_history, zhToolsDdl, downloadChk, temperatureSlider, llmDdl], 
            outputs=[chatbot, chat_history],
            show_progress='hidden'
        ).then(
            fn = lambda: [[i] for i in cache.all_files], outputs=[dstCachedPapers]
        ).then(
            fn = lambda : (
                gr.Textbox(label="用户对话框:", interactive=True, placeholder="在这里输入"),
                gr.Dropdown(interactive=True),
                gr.Slider(interactive=True)
            ), 
            inputs = None, 
            outputs = [
                txtbot,
                zhToolsDdl,
                temperatureSlider
            ]
        )
        gr.on(
            [docTxtbot.submit, docSubmitBtn.click],
            fn=chat_with_document,
            inputs=[pdfBox, docTxtbot, docChatbot],
            outputs=[docChatbot, docTxtbot]
        )
        pdfBox.change(
            check_and_clear_pdfqa_history,
            [pdfBox, docTxtbot, docChatbot],
            [docChatbot, docTxtbot]
        )
        vqaSubmitBtn.click(
            vqa_chat_submit,
            inputs=[multiModalChatbot,vqaTxtbot,vqaImgBox,mllmDdl,mllmApikeyDdl,mllmBaseurlTxt],
            outputs=[multiModalChatbot,vqaTxtbot]
        )
    return demo

# 启动Gradio界面
def main():
    load_dotenv()
    global_var._init()
    if Path(DEFAULT_CACHE_DIR).exists() == False:
        Path(DEFAULT_CACHE_DIR).mkdir(parents=True)
    
    logger.info(f'gradio version: {gr.__version__}')
    load_channel()
    init_cache()
    _init_state_vars()
    demo = create_ui()
    logger.info('SciAgent start!')
    demo.queue().launch(inbrowser=True)
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    main()
