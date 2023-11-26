
import pdfplumber
import zhipuai
import re 

def translation(text,language):
 zhipuai.api_key = "your api key"
 response = zhipuai.model_api.invoke(
    model="chatglm_lite",
    prompt=[
        {"role": "user", "content": "准确地将下面的文段翻译为"+language+",其中可能会有一些单词连在一起没有用空格分割开，你需要输出将这些单词分割后以后的翻译。你只需要在输出中包含最终对应语言的翻译："+text},
    ]
 )
 #print(response)
 return response['data']['choices'][0]['content']

def work(path_txt):
   with open(path_txt, "r", encoding="utf-8") as file:
      text = file.read()
   print("开始了！")
   abstract_pos = list(re.finditer("Abstract", text, re.IGNORECASE))
   tot_pos=0
   first_position=abstract_pos[0].start()
   tot_pos=tot_pos+first_position
   length=len(text)
   print("进度：0%")
   segments = [] 
   while tot_pos<length :   
     start_index = tot_pos  
     end_index = start_index + 2000    
     split_pos = list(re.finditer("\.", text[end_index:], re.IGNORECASE))  
     if len(split_pos)==0:
        kk=len(text)-1
     else :
        kk=split_pos[0].start()+tot_pos+2000
     if kk>length :
        kk=length-1
     segment =translation(text[start_index:kk],"中文")  
     segment =segment+"\n\n"
     TextFile2=open(path[:-4]+"_中文ver.txt",mode='a',encoding='utf-8')
     TextFile2.write(segment)
     if len(split_pos)==0:
        break
     first_position=kk+1
     tot_pos=first_position
     print("进度："+str(int(tot_pos*1.0/length*1.0*100))+"%")
   print("完成了！已经以txt形式写入路径的文件夹下。")

if __name__=='__main__':
   path=input("请输入文件路径 并注意去除路径两端的冒号")
   path_txt=path[:-4]+"_txt版.txt"
   print(path)
   with pdfplumber.open(path) as pdf:
     for page in pdf.pages:
        text=page.extract_text()
        TextFile=open(path_txt,mode='a',encoding='utf-8')
        TextFile.write(text)
   work(path_txt)

