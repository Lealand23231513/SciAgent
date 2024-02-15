# Only for test
import gradio as gr
import os
from dotenv import load_dotenv
import logging
from controler import main_for_test,main
import shutil
from pathlib import Path
from utils import DEFAULT_CACHE_DIR

def upload(file_obj):
    load_dotenv()
    cache_dir = DEFAULT_CACHE_DIR
    dst_path = shutil.copy(Path(file_obj.name), Path(cache_dir))
    logger.info('File {} uploaded successfully!'.format(os.path.basename(dst_path)))
    gr.Info('File {} uploaded successfully!'.format(os.path.basename(dst_path)))
def create_ui():
    with gr.ChatInterface(
        main,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Input here", container=False, scale=7),
        title="Sciagent",
        description="Start typing below and then click **Submit** to see the output.",
        theme="soft",
        retry_btn=None,
        clear_btn="Clear",
    ) as demo:
        with gr.Row():
            file_btn = gr.File(label="click to upload .pdf or .docx file", file_types=['.pdf','.docx'])
            file_btn.upload(upload,inputs=[file_btn])
        pass
    return demo



if __name__ == '__main__':
    print('gradio version:', gr.__version__)
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info('SciAgent start!')
    demo = create_ui()
    demo.queue().launch(share=False, inbrowser=True)