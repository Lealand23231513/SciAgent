from config import *
from state import BaseState
from typing import Optional, Any
from pydantic import field_validator, model_validator, Field, ConfigDict


class BaseModelStateConst:
    MAX_TEMPERATURE = 0.99
    MIN_TEMPERATURE = 0.01
    DEFAULT_TEMPERATURE = 0.5
    MAX_TOP_P = 0.99
    MIN_TOP_P = 0.01
    DEFAULT_TOP_P = 0.7


class BaseModelState(BaseState):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(
        default=BaseModelStateConst.DEFAULT_TEMPERATURE,
        gt=BaseModelStateConst.MIN_TEMPERATURE,
        lt=BaseModelStateConst.MAX_TEMPERATURE,
    )
    top_p: float = Field(
        default=BaseModelStateConst.DEFAULT_TOP_P,
        gt=BaseModelStateConst.MIN_TOP_P,
        lt=BaseModelStateConst.MAX_TOP_P,
    )   
    
    @model_validator(mode="after")
    def validate_environ(self):
        if self.api_key== "":
            self.api_key = None
        if self.base_url == "":
            self.base_url = None
        return self
    


class LLMStateConst(BaseModelStateConst):
    DEFAULT_LLM = SUPPORT_LLMS[0]
    LLM_CHOICES = SUPPORT_LLMS
    DEFAULT_API_KEY = None
    DEFAULT_BASE_URL = None


class LLMState(BaseModelState):
    model: str = LLMStateConst.DEFAULT_LLM
    api_key: Optional[str] = LLMStateConst.DEFAULT_API_KEY
    base_url: Optional[str] = LLMStateConst.DEFAULT_BASE_URL


class MLLMStateConst(BaseModelStateConst):
    DEFAULT_MLLM = SUPPORT_MLLMS[0]
    MLLM_CHOICES = SUPPORT_MLLMS
    DEFAULT_API_KEY = None
    DEFAULT_BASE_URL = None
    DEFAULT_MAX_TOKENS = 20
    MINIMUN_MAX_TOKENS = 20
    MAXIMUM_MAX_TOKENS = 1000


class MLLMState(BaseModelState):
    model: str = MLLMStateConst.DEFAULT_MLLM
    api_key: Optional[str] = MLLMStateConst.DEFAULT_API_KEY
    base_url: Optional[str] = MLLMStateConst.DEFAULT_BASE_URL
    max_tokens: int = Field(
        default=MLLMStateConst.DEFAULT_MAX_TOKENS,
        ge=MLLMStateConst.MINIMUN_MAX_TOKENS,
        le=MLLMStateConst.MAXIMUM_MAX_TOKENS,
    )


class EMBStateConst(BaseModelStateConst):
    DEFAULT_EMB = SUPPORT_EMBS[0]
    EMB_CHOICES = SUPPORT_EMBS
    DEFAULT_API_KEY = None
    DEFAULT_BASE_URL = None


class EMBState(BaseState):
    model: str = EMBStateConst.DEFAULT_EMB
    api_key: Optional[str] = EMBStateConst.DEFAULT_API_KEY
    base_url: Optional[str] = EMBStateConst.DEFAULT_BASE_URL

    # @field_validator('model', mode='before')
    # @classmethod
    # def validate_model(cls, value:str):
    #     if value:
    #         value = 
    #     return value