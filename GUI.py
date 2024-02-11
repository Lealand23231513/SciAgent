from json import tool
import gradio as gr
import logging
from controler import main_for_test,main
from pathlib import Path
from dotenv import load_dotenv
import os
import shutil




def add_input(user_input, chatbot, tools_ddl:list):
    chatbot.append((user_input,None))
    # tools_ddl.interactive = False
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
    # tools_ddl.interactive = True
    return chatbot, chat_history, tools_ddl

def upload(file_obj):
    load_dotenv()
    cache_dir = os.environ['DEFAULT_CACHE_DIR']
    dst_path = shutil.copy(Path(file_obj.name), Path(cache_dir))
    logger.info('File {} uploaded successfully!'.format(os.path.basename(dst_path)))

def create_ui():
    with gr.Blocks(title='SciAgent', theme='soft') as demo:  # 设置页面标题为'SciAgent'
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="SciAgent", height=900)
                txtbot = gr.Textbox(label="Your message:", placeholder="Input here")
                chat_history = gr.State([])
            with gr.Column(scale=1):
                with gr.Row():
                    tools_ddl = gr.Dropdown(
                        ["search_and_download", "communicator"],  
                        multiselect=True, 
                        label="Tools", 
                        info="Select tools to use",
                        interactive=True
                    )
                    
                with gr.Row():
                    file_btn = gr.File(label="click to upload .pdf or .docx file", file_types=['.pdf','.docx'])
                    file_btn.upload(upload,inputs=[file_btn])
        with gr.Row():
            clear_btn = gr.ClearButton([txtbot,chatbot,chat_history])
            submit_button = gr.Button("Submit")
            submit_button.click(fn = add_input, 
                                inputs = [txtbot, chatbot, tools_ddl], 
                                outputs = [txtbot, chatbot, tools_ddl]).then(
                fn = submit, inputs = [chatbot, chat_history, tools_ddl], outputs=[chatbot, chat_history, tools_ddl]
            ).then(
                fn = lambda : gr.Textbox(label="Your message:", interactive=True, placeholder="Input here"), inputs = None, outputs = [txtbot]
            )
        
        # with gr.Column():
        #     state_txt_box = gr.Textbox()
                
            
            
            

            
        # with gr.Tab('文件传输'):
        #     uploaded_file = gr.File(label="上传PDF文件", type="file")  # 允许上传PDF文件
        #     ai_response_file = gr.Textbox(label="SciAgent:",placeholder="SciAgent 将会给出概括", value='')
    return demo

# 启动Gradio界面
if __name__ == '__main__':
    print(gr.__version__)
    load_dotenv()
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info('SciAgent start!')
    demo = create_ui()
    demo.queue().launch(share=False, inbrowser=True)
