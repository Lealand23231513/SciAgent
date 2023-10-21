import gradio as gr
import pdfplumber
import openai
from controler import main_for_test,main

# 设置你的API密钥
api_key = ''
openai.api_key = api_key

# 初始化总的对话历史记录列表和当前轮对话历史记录列表
full_chat_history = []
current_chat_history = []


def submit(user_message):
    global current_chat_history
    # 将用户输入添加到当前轮对话历史记录
    current_chat_history.append(f"User: {user_message}")
    responses = ''
    for response in main(user_message):
        responses += ' '
        responses += response
    return responses


def clear():
    return ''


def history():
    full_chat = "\n".join(full_chat_history)
    return full_chat

def stock():
    global full_chat_history
    global current_chat_history
    
    # 将当前轮对话历史记录加入总的对话历史记录
    full_chat_history.append("------------------------------------------")
    full_chat_history.extend(current_chat_history)
    
    # 清空当前轮对话历史记录
    current_chat_history = []  
    return None





with gr.Blocks(title='SciAgent') as demo:  # 设置页面标题为'SciAgent'
    gr.Markdown("Start typing below and then click **Submit** to see the output.")
    with gr.Column():
        with gr.Row():
            new_chat_button = gr.Button("New Chat")
            history_chat_button = gr.Button("Chat History")
        with gr.Tab('对话'):
            user_input = gr.Textbox(label="你的信息:", placeholder="请在这里输入")
            
            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
                
            
            ai_response = gr.Textbox(label="SciAgent:", value='')
            clear_button.click(fn = clear, inputs = None, outputs=[user_input])
            submit_button.click(fn = submit, inputs = [user_input], outputs=[ai_response])
            submit_button.click(fn = clear, inputs = None, outputs=[user_input])
        
        new_chat_button.click(fn = stock, inputs = None, outputs = None)
        new_chat_button.click(fn = clear, inputs = None, outputs = [user_input])
        new_chat_button.click(fn = clear, inputs = None, outputs = [ai_response])
        history_chat_button.click(fn = history, inputs = None, outputs = [ai_response])
            
            
        with gr.Tab('文件传输'):
            uploaded_file = gr.File(label="上传PDF文件", type="file")  # 允许上传PDF文件
            ai_response_file = gr.Textbox(label="SciAgent:",placeholder="SciAgent 将会给出概括", value='')


# 启动Gradio界面
demo.queue().launch(share=False, inbrowser=True)
