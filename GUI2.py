import gradio as gr
import logging
from controler import main_for_test,main
from pathlib import Path

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
    pass
if __name__ == '__main__':
    print(gr.__version__)
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(Path(__file__).stem)
    logger.info('SciAgent start!')
    x = demo.queue().launch(share=False, inbrowser=True)