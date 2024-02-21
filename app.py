import gradio as gr
import logging
from controler import main_for_test,main, call_agent
from pathlib import Path
from dotenv import load_dotenv
from Retrieval_qa import Cache
import os
from utils import DEFAULT_CACHE_DIR, TOOLS_LIST
from typing import cast
from ws_server import load_ws_server
import global_var
from cache import load_cache

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
    tools = [
        {
            "name": tool,
            "kwargs":{
                "download":downloadChkValue
            }
        }
        for tool in tools_ddl
    ]
    for ai_response in call_agent(user_input, chat_history, tools=tools, stream=True):
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
    cache.cache_file(str(Path(file_obj.name)))
    gr.Info('File {} uploaded successfully!'.format(os.path.basename(Path(file_obj.name))))

    return [[i] for i in cache.all_files]

def create_ui():
    html = None
    with open('websocket.html',encoding='utf-8') as f:
        html = f.read()
    with gr.Blocks(title='SciAgent', theme='soft', head=html) as demo:
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
                    with gr.Group():
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
            fn = submit, inputs = [chatbot, chat_history, toolsDdl, downloadChk], outputs=[chatbot, chat_history, toolsDdl, downloadChk]
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
    logger.info(f'gradio version:{gr.__version__}')
    server = load_ws_server()
    cache = load_cache()
    demo = create_ui()
    logger.info('SciAgent start!')
    demo.queue().launch(share=False, inbrowser=True)
