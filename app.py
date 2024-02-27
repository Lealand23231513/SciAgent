import gradio as gr
import logging
from controler import main_for_test,main, call_agent
from pathlib import Path
from dotenv import load_dotenv
import os
from utils import DEFAULT_CACHE_DIR, TOOLS_LIST
from gradio_modal import Modal
from typing import cast
import global_var
from cache import load_cache
from channel import load_channel
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
    gr.Info(f'All cached files are cleaned.')
    return []

def add_input(user_input, chatbot, tools_ddl:list):
    chatbot.append((user_input,None))
    return (                                                                                                                                                                                                        
            gr.Textbox(interactive = False, value = '', placeholder = ''), 
            chatbot, 
            tools_ddl
            )

def submit(chatbot, chat_history, tools_ddl:list, downloadChkValue:bool):
    user_input = chatbot[-1][0]
    chat_history.append(
        {
            "role": "user",
            "content": f"{user_input}"
        }
    )
    full_response = ""
    tools = []
    for tool in tools_ddl:
        config = {
            "name":tool,
            "kwargs": {}
        }
        if tool=='websearch':
            config['kwargs']['download']=downloadChkValue
        tools.append(config)
    for ai_response in call_agent(user_input, chat_history, model='gpt-3.5-turbo', tools=tools, stream=True):
        full_response += str(ai_response)    #type: ignore
        chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot, chat_history, tools_ddl, downloadChkValue
    chat_history.append(
        {
            "role": "assistant",
            "content": f"{full_response}"
        }
    )
    logger.info("submit end")
    return chatbot, chat_history, tools_ddl, downloadChkValue

def upload(file_obj):
    cache = load_cache()
    cache.cache_file(str(Path(file_obj.name)), save_local=True)
    gr.Info('File {} uploaded successfully!'.format(os.path.basename(Path(file_obj.name))))

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
    html = None
    with open('websocket.html',encoding='utf-8') as f:
        html = f.read()
    with gr.Blocks(title='SciAgent', theme='soft') as demo:
        with gr.Tab(label='chat'):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SciAgent", height=900)
                    txtbot = gr.Textbox(label="Your message:", placeholder="Input here", lines=4)
                    chat_history = gr.State([])
                with gr.Column(scale=1):
                    with gr.Row():
                        toolsDdl = gr.Dropdown(
                            cast(list[str | int | float | tuple[str, str | int | float]] | None, TOOLS_LIST),  
                            multiselect=True, 
                            label="Tools", 
                            info="Select tools to use",
                            interactive=True
                        )
                        
                    with gr.Row():
                        uploadFileBtn = gr.File(
                            label="click to upload .pdf or .docx file", 
                            file_types=['.pdf','.docx']
                        )
                        
                    with gr.Row():
                        cleanCacheBtn = gr.Button('Clear all cached files')
                    with gr.Row():
                        llmDdl = gr.Dropdown(
                            choices=['gpt-3.5-turbo','chatglm3'],
                            value=['gpt-3.5-turbo'],
                            label="Model"
                        )
                    with gr.Accordion(label='websearch params', open=False):
                        downloadChk = gr.Checkbox(
                            label='download',
                        )
                        
            with gr.Row():
                clearBtn = gr.ClearButton([txtbot,chatbot,chat_history])
                submitBtn = gr.Button("Submit")
                
        with gr.Tab(label='Cached papers'):
            cache = load_cache()
            dstCachedPapers = gr.Dataset(
                components=[gr.Textbox(visible=False)], 
                label='Cached papers',
                samples=[[i] for i in cache.all_files]
            )
        with gr.Tab(label='log'):
            pass
        # with gr.Column():
        #     state_txt_box = gr.Textbox()
        timeStampDisp = gr.Textbox(label='time stamp', value=get_timestamp, every=1, visible=False)
        with Modal(visible=False) as modal:
            modalMsg = gr.Markdown()
            with gr.Row():
                confirmBtn = gr.Button("Yes")
                cancelBtn = gr.Button("No")
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
            inputs = [txtbot, chatbot, toolsDdl], 
            outputs = [txtbot, chatbot, toolsDdl]
        ).then(
            fn = submit, 
            inputs = [chatbot, chat_history, toolsDdl, downloadChk], 
            outputs=[chatbot, chat_history, toolsDdl, downloadChk],
            show_progress='hidden'
        ).then(
            fn = lambda: [[i] for i in cache.all_files], outputs=[dstCachedPapers]
        ).then(
            fn = lambda : gr.Textbox(label="Your message:", interactive=True, placeholder="Input here"), inputs = None, outputs = [txtbot]
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
