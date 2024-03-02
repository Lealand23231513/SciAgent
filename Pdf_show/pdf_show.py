#需提前部署gradio_pdf poppler pytesseract tesseractOCR ;第一个影响页面部署 后三个影响pdf识别
#下面用的这个模型效果不是很理想...
import gradio as gr
from gradio_pdf import PDF
from pdf2image import convert_from_path
from transformers import pipeline
from pathlib import Path

__file__='输入文件地址 如C:/example/example1.pdf'
dir_ = Path(__file__).parent

p = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

def fn(question: str, doc: str) -> str:
    img = convert_from_path(doc)[0]
    output = p(img, question)
    return sorted(output, key=lambda x: x["score"], reverse=1000)[0]['answer']


demo = gr.Interface(
    fn,
    [gr.Textbox(label="Question"), PDF(height=1000,label="Document",min_width=300)],
    gr.Textbox(),
    examples=[["Who is the first writer of the paper?", str(dir_ / "example1.pdf")],
              ["What is the biggest advantage of the new model?", str(dir_ / "example2.pdf")]]
)

if __name__ == "__main__":
    demo.launch()