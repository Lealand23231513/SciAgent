import gradio as gr
import pdfplumber

# 初始化对话历史记录列表
chat_history = []

# 定义一个函数，用于回应用户的消息
def chat_with_ai(user_message, uploaded_file, chat_option):
    
    # 初始化回应
    ai_response = "AI: This is an example."

    # 处理上传的PDF文件
    if uploaded_file is not None:
        try:
            pdf_path = uploaded_file.value
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text()
            # 在这里处理PDF文件文本，可以根据需要修改回应
        except Exception as e:
            ai_response = f"AI: An error occurred while processing the PDF file: {str(e)}"
    
    # 将用户输入和AI回应添加到历史记录
    chat_history.append(f"User: {user_message}")
    chat_history.append(f"AI:{ai_response}")

    # 根据下拉选项，决定显示的对话记录
    if chat_option == "新的对话":
        ai_response = ""
    elif chat_option == "历史记录":
        full_chat = "\n".join(chat_history)
        ai_response = full_chat

    return ai_response


def submit(user_message):
    ai_response = gr.Textbox(label="SciAgent:", value=chat_with_ai(user_message,None,chat_option)) 
    return ai_response



with gr.Blocks(title='SciAgent') as demo:  # 设置页面标题为'SciAgent'
    with gr.Column():
        with gr.Row():
            chat_option = gr.Button("新的对话")
            chat_option = gr.Button("历史记录")
        with gr.Tab('对话'):
            user_input = gr.Textbox(label="你的信息:", placeholder="请在这里输入")
            with gr.Row():
                clear_button = gr.Button("清空")
                submit_button = gr.Button("提交")
                
                
            ai_response = gr.Textbox(label="SciAgent:", value='')
            
            submit_button.click(submit, inputs = [user_input], outputs=[ai_response])
            
            

            
            
            
        with gr.Tab('文件传输'):
            uploaded_file = gr.File(label="上传PDF文件", type="file")  # 允许上传PDF文件
            ai_response_file = gr.Textbox(label="SciAgent:",placeholder="SciAgent 将会给出概括", value='')


# 启动Gradio界面
demo.queue().launch(share=False, inbrowser=True)


#问题：Button“清空”功能没实现，chat_option不知道怎么设计逻辑用到ai_response里面,Button“提交”返回的不是我设定的默认值（大概是系统默认的）