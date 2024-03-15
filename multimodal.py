import logging
from httpx import HTTPError
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import Stream
from typing import Optional, cast
from pathlib import Path
import base64

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
    model: str,
    stream=False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    if api_key=='':
        api_key=None
    if base_url=='':
        base_url=None
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(user_message)
    logger = logging.getLogger(Path(__file__).name)
    logger.info(messages)
    try:
        response = client.chat.completions.create(
            model=model,
            stream=stream,
            messages=messages,
            max_tokens=10
        )
    except HTTPError as httpe:
        logger.exception(httpe)
    if stream:
        response = cast(Stream[ChatCompletionChunk], response)
        for chunk in response:
            part = chunk.choices[0].delta.content
            if part:
                yield part
    else:
        response = cast(ChatCompletion, response)
        return response.choices[0].message.content
