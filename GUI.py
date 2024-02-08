import gradio as gr
import logging
from controler import main_for_test,main
from pathlib import Path
from dotenv import load_dotenv




def add_input(user_input, chatbot):
    chatbot.append((user_input,""))
    return gr.Textbox(interactive = False, value = '', placeholder = ''), chatbot

def submit(chatbot, chat_history):
    user_input = chatbot[-1][0]
    chat_history.append(
        {
            "role": "user",
            "content": f"{user_input}"
        }
    )
    full_response = ""
    for ai_response in main(user_input, chat_history):
        chat_history += [
            {
                "role": "assistant",
                "content": f"{ai_response}"
            }
        ]
        full_response += ai_response +'\n'
        chatbot[-1] = (chatbot[-1][0], full_response)
        yield chatbot, chat_history
    logger.info("submit end")


def clear():
    return '', [], []

with gr.Blocks(title='SciAgent') as demo:  # 设置页面标题为'SciAgent'
    gr.Markdown("Start typing below and then click **Submit** to see the output.")
    with gr.Column():

        with gr.Tab('对话'):
            chatbot = gr.Chatbot(label="SciAgent")
            user_input = gr.Textbox(label="你的信息:", placeholder="请在这里输入")
            chat_history = gr.State([])
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
                
            
            clear_button.click(fn = clear, inputs = None, outputs=[user_input, chatbot, chat_history])
            submit_button.click(fn = add_input, inputs=[user_input, chatbot], outputs = [user_input, chatbot], queue = False).then(
                fn = submit, inputs = [chatbot, chat_history], outputs=[chatbot, chat_history]
            ).then(
                fn = lambda : gr.Textbox(label="你的信息:", interactive=True, placeholder="请在这里输入"), inputs = None, outputs = [user_input], queue = False
            )
        
        # with gr.Tab('文件传输'):
        #     uploaded_file = gr.File(label="上传PDF文件", type="file")  # 允许上传PDF文件
        #     ai_response_file = gr.Textbox(label="SciAgent:",placeholder="SciAgent 将会给出概括", value='')


# 启动Gradio界面
if __name__ == '__main__':
    print(gr.__version__)
    load_dotenv()
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info('SciAgent start!')
    x = demo.queue().launch(share=False, inbrowser=True)
