#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install zhipuai


# In[81]:


#import zhipuai


# In[83]:


def translation(text,language):
 zhipuai.api_key = "8670e5d6974d32926cf0b5a5be87bd6e.LkKvB58YqYCrIWGr"
 response = zhipuai.model_api.invoke(
    model="chatglm_lite",
    prompt=[
        {"role": "user", "content": "准确地将下面的文段翻译为"+language+",你只需要在输出中包含最终对应语言的翻译："+text},
    ]
 )
 #print(response)
 print((response['data']['choices'][0]['content']))


# In[85]:


language=input("请输入需要翻译到的语种")
text=input("请输入需要翻译的部分")
translation(text,language)


# In[ ]:





# In[ ]:




