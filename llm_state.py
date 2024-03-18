from config import *
from state import BaseState
from typing import Optional
from pydantic import model_validator

class LLMConst():
    DEFAULT_LLM=SUPPORT_LLMS[0]
    LLM_CHOICES=SUPPORT_LLMS

class LLMState(BaseState):
    model: str = LLMConst.DEFAULT_LLM
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    @model_validator(mode="after")
    def validate_environ(cls, values):
        if values.model not in SUPPORT_LLMS:
            raise ValueError(f'llm {values.model} is not support.')
        if values.api_key == "":
            values.api_key = None
        if values.base_url == "":
            values.base_url = None
        return values
    