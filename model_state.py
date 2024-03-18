from config import *
from state import BaseState
from typing import Optional
from pydantic import model_validator, Field


class LLMStateConst:
    DEFAULT_LLM = SUPPORT_LLMS[0]
    LLM_CHOICES = SUPPORT_LLMS
    DEFAULT_API_KEY = None
    DEFAULT_BASE_URL = None


class LLMState(BaseState):
    model: str = LLMStateConst.DEFAULT_LLM
    api_key: Optional[str] = LLMStateConst.DEFAULT_API_KEY
    base_url: Optional[str] = LLMStateConst.DEFAULT_BASE_URL

    # @model_validator(mode="after")
    # def validate_environ(cls, values):
    #     if values.model not in LLMStateConst.LLM_CHOICES:
    #         raise ValueError(f"llm {values.model} is not support.")
    #     if values.api_key == "":
    #         values.api_key = None
    #     if values.base_url == "":
    #         values.base_url = None
    #     return values


class MLLMStateConst:
    DEFAULT_MLLM = SUPPORT_MLLMS[0]
    MLLM_CHOICES = SUPPORT_MLLMS
    DEFAULT_API_KEY = None
    DEFAULT_BASE_URL = None
    DEFAULT_MAX_TOKENS = 20
    MINIMUN_MAX_TOKENS = 20
    MAXIMUM_MAX_TOKENS = 1000


class MLLMState(BaseState):
    model: str = MLLMStateConst.DEFAULT_MLLM
    api_key: Optional[str] = MLLMStateConst.DEFAULT_API_KEY
    base_url: Optional[str] = MLLMStateConst.DEFAULT_BASE_URL
    max_tokens: int = Field(
        default=MLLMStateConst.DEFAULT_MAX_TOKENS,
        le=MLLMStateConst.MINIMUN_MAX_TOKENS,
        ge=MLLMStateConst.MAXIMUM_MAX_TOKENS,
    )
    # @model_validator(mode="after")
    # @classmethod
    # def validate_environ(cls, values):
    #     if values.model not in MLLMStateConst.MLLM_CHOICES:
    #         raise ValueError(f"llm {values.model} is not support.")
    #     if values.api_key == "":
    #         values.api_key = None
    #     if values.base_url == "":
    #         values.base_url = None
    #     return values