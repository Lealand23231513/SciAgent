import zhipuai
def translation(text,language):
    zhipuai.api_key = "your api key"
    response = zhipuai.model_api.invoke(
    model="chatglm_lite",
    prompt=[
        {"role": "user", "content": "准确地将下面的文段翻译为"+language+",你只需要在输出中包含最终对应语言的翻译："+text},
    ]
    )
    #print(response)
    print((response['data']['choices'][0]['content']))
language=input("请输入需要翻译到的语种")
text=input("请输入需要翻译的部分")
translation(text,language)