import logging
from httpx import HTTPError
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import Stream
from typing import Optional, cast
from pathlib import Path
import base64

from global_var import get_global_value
from model_state import MLLMState

logger = logging.getLogger(Path(__file__).stem)


def _encode_image(image_path: str | Path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _build_messages(user_message: list[str | dict[str, str]]) -> list:
    content = []
    for m in user_message:
        if isinstance(m, str):
            content.append({"type": "text", "text": m})
        elif isinstance(m, dict):
            if m["type"] == "file":
                base64_img = _encode_image(m["filepath"])
                extension = Path(m["filepath"]).suffix
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{extension};base64,{base64_img}"
                        },
                    }
                )
            elif m["type"] == "text":
                content.append({"type": "text", "text": m["text"]})
            else:
                raise ValueError("Wrong user_messagge key")
    if len(content)==0:
        raise ValueError("No content of user_message received")
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    return messages

def multimodal_chat(
    user_message: list,
):
    mllm_state = cast(MLLMState, get_global_value('mllm_state'))
    
    api_key=mllm_state.api_key if mllm_state.api_key else None
    base_url = mllm_state.base_url if mllm_state.base_url else None
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(user_message)
    logger = logging.getLogger(Path(__file__).name)
    logger.info(mllm_state.model_dump())
    try:
        response = client.chat.completions.create(
            model=mllm_state.model,
            messages=messages,
            max_tokens=mllm_state.max_tokens
        )
        response = cast(ChatCompletion, response)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(e)
        return repr(e)

def multimodal_chat_stream(
    user_message: list,
):
    mllm_state = cast(MLLMState, get_global_value('mllm_state'))
    
    api_key=mllm_state.api_key if mllm_state.api_key else None
    base_url = mllm_state.base_url if mllm_state.base_url else None
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(user_message)
    logger = logging.getLogger(Path(__file__).name)
    logger.info(mllm_state.model_dump())
    try:
        response = client.chat.completions.create(
            model=mllm_state.model,
            stream=True,
            messages=messages,
            max_tokens=mllm_state.max_tokens
        )
        response = cast(Stream[ChatCompletionChunk], response)
        for chunk in response:
            part = chunk.choices[0].delta.content
            if part:
                yield part
    except Exception as e:
        logger.error(e)
        yield repr(e)
