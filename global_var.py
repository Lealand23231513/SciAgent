# -*- coding: utf-8 -*-
import logging
def _init():  # 初始化
    global _global_dict
    _global_dict = {}

def set_global_value(key, value):
    #定义一个全局变量
    _global_dict[key] = value

def get_global_value(key:str):
    #获得一个全局变量，不存在则提示读取对应变量失败
    return _global_dict.get(key)