# -*- coding: utf-8 -*-
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _init():
    global _global_dict, _waiting_dict
    _global_dict, _waiting_dict = {}, {}
    


def set_global_value(key: str, value: Any):
    """Define a global var"""
    _global_dict[key] = value


def get_global_value(key: str)-> Any:
    """get global var"""
    return _global_dict.get(key)
