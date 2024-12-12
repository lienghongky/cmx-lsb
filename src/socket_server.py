# websocket_server.py
import asyncio
import websockets
from typing import List

class WebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.connections: List[websockets.WebSocketServerProtocol] = []

    async def handler(self, websocket):
        # Add the connection to the list of active connections
        self.connections.append(websocket)
        try:
            while True:
                # Wait for a message from the client (optional)
                message = await websocket.recv()
                print(f"Received message: {message}")
        except websockets.ConnectionClosed:
            pass
        finally:
            # Remove the connection when closed
            self.connections.remove(websocket)

    async def emit(self, data: str):
        # Send data to all connected clients
        # print(f"Sending data: {data}")
        if self.connections:
            await asyncio.wait([conn.send(data) for conn in self.connections])

# Singleton WebSocket server instance
ws_server = WebSocketServer()

# Start the WebSocket server
async def start_server():
    server = await websockets.serve(ws_server.handler, ws_server.host, ws_server.port)
    print(f"WebSocket server started on ws://{ws_server.host}:{ws_server.port}")
    await server.wait_closed()

# Start the WebSocket server in a background task
async def run_websocket_server():
    await start_server()
