from bs4 import BeautifulSoup
from fake_useragent import FakeUserAgent
import threading
import os
import requests
import urllib.request
from pathlib import Path
import re
import time
import openai
import json
from urllib import parse


def google_scholar_auto_search_and_download(query: str, download: bool = True, top_k_result=3) -> list[dict[str, str]] | str | None:

    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    headers = {
        'User-Agent': FakeUserAgent().random
    }
    min_citation_count = 500  # 默认设置最少引用数量为500

    prompt = f"""请你将该单词翻译成英文，并且只返回翻译后的英文单词：{query}"""
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    query = response.choices[0].message.content

    keyword = re.sub(" +", "+", query.strip())
    url_base = 'https://scholar.google.com/scholar?hl=en&as_sdt=0'
    url_base = url_base + '&q=' + keyword

    count = 0
    start = 0

    paper_list = []

    while(count < top_k_result) & (start < 200):

        time.sleep(3)

        url = url_base + '&start=' + str(start)
        start = start + 10

        count_0 = count

        req = urllib.request.Request(url=url, headers=headers)
        res = urllib.request.urlopen(req, timeout=100)

        html = res.read().decode('utf-8')
        soup = BeautifulSoup(html, 'lxml')

        try:
            for div in soup.select('.gs_or'):
                if len(div.select('.gs_fl > a')[2].string) < 10:
                    continue
                citation_count = int(div.select('.gs_fl > a')[2].string[9:])
                if citation_count < min_citation_count:
                    continue
                else:
                    url = div.find('a')['href']
                    if url.endswith(".pdf"):
                        paper = {
                            "link": url,
                            "title": div.select('.gs_rt > a')[0].text,
                            "citation count": str(citation_count)
                        }
                        paper_list.extend([paper])
                        count = count + 1
                    else:
                        continue
                if count >= top_k_result:
                    break
        except Exception as e:
            break

    if download == False:
        return paper_list
    folder_name = query
    folder_name = parse.quote(folder_name, safe = "")
    same_name_cnt = 1

    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        while(os.path.exists(folder_name+f"({str(same_name_cnt)})")):
            same_name_cnt += 1
        folder_name += f"({str(same_name_cnt)})"
        print(f"当前路径下已存在同名文件夹，故创建新文件夹\"{folder_name}\"")

    for paper in paper_list:
        url = paper["link"]  
        title = paper["title"]  
        citation_count = paper["citation count"]
        title = parse.quote(title, safe = "")

        res = requests.get(url)
        if res.status_code == 200:

            file_name = os.path.join(folder_name, title + '.pdf')

            with open(file_name, 'wb') as pdf_file:
                pdf_file.write(res.content)
            print(f'文件 {title}.pdf 下载成功！')
        else:
            print(f'下载失败, HTTP状态码: {res.status_code}')

    return paper_list


def search_and_download(user_input: str):

    openai.api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI()
    messages = [{"role": "user", "content": f"{user_input}"}]

    with open('module.json', "r") as f:
        module_descriptions = json.load(f)

    functions = module_descriptions[0]["functions"]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        temperature=0,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response.choices[0].message

    if response_message.function_call:
        function_args = json.loads(response_message.function_call.arguments)
        print(function_args)
        arg_download = function_args.get("download")
        arg_top_k_results = function_args.get("top_k_results")
        google_scholar_result = google_scholar_auto_search_and_download(
            query=function_args.get("query"),
            download=arg_download if arg_download is not None else False,
            top_k_result=arg_top_k_results if arg_top_k_results is not None else 3
        )
        return google_scholar_result
    else:
        return None