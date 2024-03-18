from queue import Queue
from typing import Literal, cast
import time
import json
from global_var import *
def load_channel():
    channel = cast(Channel, get_global_value('channel'))
    if channel is None:
        channel = Channel()
        set_global_value('channel', channel)
    return channel
class Channel(object):
    def __init__(self) -> None:
        self.qBack2Front = Queue()
        self.qFront2Back = Queue()
    def send(self, message:str, this:Literal['front','back']):
        if this=='front':
            self.qFront2Back.put(message)
        else:
            self.qBack2Front.put(message)
    def recv(self, this:Literal['front', 'back']) -> str|None:
        if this=='front':
            if self.qBack2Front.empty():
                return None
            return self.qBack2Front.get()
        else:
            if self.qFront2Back.empty():
                return None
            return self.qFront2Back.get()
    def push(self, msg:str, delay=0.2, require_response = False) -> str|None:
        self.send(msg, 'back')
        if require_response == False:
            return None
        response = self.recv('back')
        while response is None:
            response = self.recv('back')
            time.sleep(delay)
        return response
    def show_modal(self, name:Literal['info','error','warning'], message:str):
        msg = json.dumps(
            {
                "type": "modal",
                "name": name,
                "message": message,
            }
        )
        self.send(msg, this='back')