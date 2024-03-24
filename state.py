import logging
import global_var
from global_var import *
from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, cast, Callable, Optional
from config import *


class BaseState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    def set_global(self, global_name: str):
        set_global_value(global_name, self)

    def change(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class StateMutex(object):
    """
    a decorator
    """

    def __init__(self, unlock_after_func: bool = True):
        self.unlock_after_func = unlock_after_func

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            logger.debug("set state_mutex")
            self.set_state_mutex(True)
            res = func(*args, **kwargs)
            if self.unlock_after_func:
                self.set_state_mutex(False)
                logger.debug("unset state_mutex")
            return res

        return wrapper
    @classmethod
    def set_state_mutex(cls, lock: bool):
        _old_state_mutex = get_global_value("state_mutex")
        set_global_value("state_mutex", lock)
        if lock == False and _old_state_mutex == True:
            for k in global_var._waiting_dict:
                set_global_value(k, global_var._waiting_dict[k])
            global_var._waiting_dict = {}


def change_state(name: str, **kwargs):
    """
    :param name: name of the gradio state var
    """
    logger.debug(kwargs)
    print(kwargs)
    if get_global_value(name):
        curr_state = get_global_value(name)
        if isinstance(curr_state, BaseState) == False:
            raise ValueError(
                f"global var {name} is not an instance of BaseState or a subclass of BaseState"
            )
        curr_state = cast(BaseState, curr_state)
        if get_global_value("state_mutex"):
            new_state = curr_state.model_copy(deep=True)
            new_state.change(**kwargs)
            global_var._waiting_dict[name] = new_state
        else:
            global_var._global_dict[name].change(**kwargs)
    else:
        raise ValueError(f"global var {name} not exist")
