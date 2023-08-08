from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    Request
)
import os, string, json
from typing import List

class SocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data):
        for connection in self.active_connections:
            await connection.send_json(data) 
            
    # async def stream_broadcast(self, data):
    #     for connection in self.active_connections:
    #         data_stream = iter(data)
    #         try:
    #             while True:
    #                 asyncio.sleep(0.1)
    #                 await connection.send_json(next(data)) 
    #         except:
    #             continue
           