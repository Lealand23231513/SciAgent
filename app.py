import gradio as gr
import logging
from controler import call_agent
from pathlib import Path
from dotenv import load_dotenv
import os
from utils import DEFAULT_CACHE_DIR, TOOLS_LIST, SUPPORT_LLMS, toolname_zh2en
from gradio_modal import Modal
from typing import cast
import global_var
from cache import load_cache
from channel import load_channel
from audio import wav2txt_client
import json
import time




def _init_state_vars():
    global state, timestamp
    state = {
        "type": None,
        "name": None,
        "message": None
    }
    timestamp = str(time.time())

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

def submit(
        chatbot, 
        chat_history, 
        zhToolsDdl:list, 
        downloadChkValue:bool, 
        temperatureValue:float,
        llmDdlValue:str
    ):
    enToolsDdl = toolname_zh2en(zhToolsDdl)
    user_input = chatbot[-1][0]
    chat_history.append(
        {
            "role": "user",
            "content": f"{user_input}"
        }
    )
    full_response = ""
    tools = []
    for tool in enToolsDdl:
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
    channel.send(response_msg, 'front')
    return Modal(visible=False)
def get_timestamp():
    global timestamp, state
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
                    audio = gr.Audio(sources="microphone", type="filepath")
                    txtbot = wav2txt_client(wav_path=audio)
                    chat_history = gr.State([])
                    with gr.Row():
                        clearBtn = gr.ClearButton(
                            value = "清除",
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
                            label="点击以下载 .pdf 或者 .docx 文件", 
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
                        # embDdl = gr.Dropdown(
                        #     choices=cast(list[str | int | float | tuple[str, str | int | float]] | None, SUPPORT_EMBS),
                        #     value="text-embedding-ada-002",
                        #     label="Embedding模型"
                        # )
                    with gr.Accordion(label='联网搜索设置', open=False):
                        downloadChk = gr.Checkbox(
                            label='下载',
                        )
                    with gr.Accordion(label='检索增强搜索（RAG）设置', open=False):
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
    return demo

# 启动Gradio界面
if __name__ == '__main__':
    load_dotenv()
    global_var._init()
    if os.path.exists(Path(DEFAULT_CACHE_DIR)) == False:
        os.mkdir(Path(DEFAULT_CACHE_DIR))
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info(f'gradio version: {gr.__version__}')
    # server = load_ws_server()
    channel = load_channel()
    cache = load_cache()
    _init_state_vars()
    demo = create_ui()
    logger.info('SciAgent start!')
    demo.queue().launch(share=False, inbrowser=True)
