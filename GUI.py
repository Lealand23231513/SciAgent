import gradio as gr
import logging
from controler import main_for_test,main
from pathlib import Path
from dotenv import load_dotenv
from Retrieval_qa import Cache
import os
from utils import DEFAULT_CACHE_DIR, TOOLS_LIST
from typing import cast

def clear_cache(dstState:Cache):
    dstState.clear_all()
    gr.Info(f'All cached files are cleaned.')
    return [], dstState



def add_input(user_input, chatbot, tools_ddl:list):
    chatbot.append((user_input,None))
    return (                                                                                                                                                                                                        
            gr.Textbox(interactive = False, value = '', placeholder = ''), 
            chatbot, 
            tools_ddl
            )

def submit(chatbot, chat_history, tools_ddl:list):
    user_input = chatbot[-1][0]
    chat_history.append(
        {
            "role": "user",
            "content": f"{user_input}"
        }
    )
    full_response = ""
    for ai_response in main(user_input, chat_history, tools=tools_ddl, stream=True):
        full_response += str(ai_response)    #type: ignore
        chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot, chat_history, tools_ddl
    chat_history.append(
        {
            "role": "assistant",
            "content": f"{full_response}"
        }
    )
    logger.info("submit end")
    return chatbot, chat_history, tools_ddl

def upload(file_obj, dstState:Cache):
    dstState.cache_file(str(Path(file_obj.name)))
    gr.Info('File {} uploaded successfully!'.format(os.path.basename(Path(file_obj.name))))

    return [[i] for i in dstState.all_files], dstState

def create_ui():
    with gr.Blocks(title='SciAgent', theme='soft') as demo:
        with gr.Tab(label='chat'):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="SciAgent", height=900)
                    txtbot = gr.Textbox(label="Your message:", placeholder="Input here")
                    chat_history = gr.State([])
                with gr.Column(scale=1):
                    with gr.Row():
                        tools_ddl = gr.Dropdown(
                            cast(list[str | int | float | tuple[str, str | int | float]] | None, TOOLS_LIST),  
                            multiselect=True, 
                            label="Tools", 
                            info="Select tools to use",
                            interactive=True
                        )
                        
                    with gr.Row():
                        file_btn = gr.File(label="click to upload .pdf or .docx file", file_types=['.pdf','.docx'])
                        
                    with gr.Row():
                        cl_cache_btn = gr.Button('Clear all cached files')
                        
            with gr.Row():
                clear_btn = gr.ClearButton([txtbot,chatbot,chat_history])
                submit_btn = gr.Button("Submit")
                
        with gr.Tab(label='Cached papers'):
            dstState = gr.State(Cache())
            dstCachedPapers = gr.Dataset(
                components=[gr.Textbox(visible=False)], label='Cached papers',
                samples=[[i] for i in dstState.value.all_files]
                )
        with gr.Tab(label='log'):
            pass
        # with gr.Column():
        #     state_txt_box = gr.Textbox()
        file_btn.upload(
            upload,
            inputs=[file_btn, dstState], 
            outputs=[dstCachedPapers, dstState],
            queue=True)
            
        cl_cache_btn.click(
            clear_cache,
            inputs=[dstState],
            outputs=[dstCachedPapers, dstState])

        submit_btn.click(
            fn = add_input, 
            inputs = [txtbot, chatbot, tools_ddl], 
            outputs = [txtbot, chatbot, tools_ddl]
        ).then(
            fn = submit, inputs = [chatbot, chat_history, tools_ddl], outputs=[chatbot, chat_history, tools_ddl]
        ).then(
            fn = lambda : gr.Textbox(label="Your message:", interactive=True, placeholder="Input here"), inputs = None, outputs = [txtbot]
        )



            

            
    
    return demo

# 启动Gradio界面
if __name__ == '__main__':
    print(gr.__version__)
    load_dotenv()
    if os.path.exists(Path(DEFAULT_CACHE_DIR)) == False:
        os.mkdir(Path(DEFAULT_CACHE_DIR))
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info('SciAgent start!')
    demo = create_ui()
    demo.queue().launch(share=False, inbrowser=True)
