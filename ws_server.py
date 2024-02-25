import asyncio
import websockets
import threading
from threading import Thread
from functools import partial
import json
from queue import Queue
from pathlib import Path
import logging

from global_var import get_global_value, set_global_value
from typing import cast

logger = logging.getLogger(Path(__file__).stem)
def load_ws_server():
    server = cast(WebSocketServer, get_global_value('ws_server'))
    if server is None:
        server = WebSocketServer()
        set_global_value('ws_server', server)
    return server
    
class WebSocketServer(object):
    #TODO: How to stop the server?
    def __init__(self, host="localhost", port=5001) -> None:
        self.host = host
        self.port = port
        self.q_in = Queue()
        self.q_out = Queue()
        self.event = threading.Event()
        self.server_thread = Thread(target=self.subThread,kwargs={"q_in":self.q_in, "q_out":self.q_out, "event":self.event})
        self.server_thread.start()
        logger = logging.getLogger(Path(__file__).stem)
        logger.info(f'start websocket at ws://{self.host}:{self.port}')
    
    async def run(self, websocket, q_in:Queue, q_out:Queue, event:threading.Event):
        while True:
            if q_in.empty() == False:
                msg = q_in.get()
                await websocket.send(msg)
                logger.debug(f"send>>> {msg}")
                # print(f"send>>> {msg}")
            res = await websocket.recv()
            logger.debug(f"recv<<< {res}")
            # print(f"recv<<< {res}")
            if res=='check':
                await websocket.send('check')
            else:
                res = json.loads(res)
                if res['a']=='check':
                    await websocket.send('check')
                else:
                    q_out.put(res)
                    event.set()

    async def core(self,**kwargs):
        bind_deal = partial(self.run, **kwargs)
        async with websockets.serve(bind_deal, self.host, self.port) as start_server:# type: ignore
            await start_server.wait_closed()
    def subThread(self,**kwargs):
        asyncio.run(self.core(**kwargs))
    def contact(self, msg:str):
        self.q_in.put(msg)
        self.event.clear()
        self.event.wait()
        res = self.q_out.get()
        logger.debug(f"main Thread recv {res}")
        return res
    def contacts(self, msgs:list[str]):
        for msg in msgs:
            self.q_in.put(msg)
        self.event.clear()
        self.event.wait()
        recvs = []
        while self.q_out.empty() == False:
            res = self.q_out.get()
            logger.debug(f"main Thread recv {res}")
            recvs.append(res)
        return recvs
            
def main():
    server = WebSocketServer()
if __name__=='__main__':
    main()