import asyncio
import json
from fastapi import FastAPI, WebSocket
from your_gemini_library import GeminiLiveClient # Hypothesized client

app = FastAPI()

# Configuration for your separate TTS microservice
TTS_WS_URL = "ws://tts-service-endpoint/v1/stream"

async def stream_to_tts(text_iterator, client_websocket):
    """Sends buffered text to TTS and forwards audio to user."""
    async with websockets.connect(TTS_WS_URL) as tts_ws:
        buffer = ""
        
        async for text_token in text_iterator:
            buffer += text_token
            
            # Send to TTS when a natural pause (sentence/phrase) occurs
            if any(punct in text_token for punct in [".", "!", "?", "\n"]):
                await tts_ws.send(json.dumps({"text": buffer.strip()}))
                buffer = ""
                
                # Immediately catch the audio chunk and send to user
                audio_chunk = await tts_ws.recv()
                await client_websocket.send_bytes(audio_chunk)

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize Gemini Live API
    gemini = GeminiLiveClient(api_key="YOUR_KEY")
    
    async with gemini.session() as session:
        while True:
            # 1. Receive User Audio from Frontend
            user_audio = await websocket.receive_bytes()
            
            # 2. Feed to Gemini (STT + LLM happens here)
            # We assume Gemini returns an async iterator of text tokens
            text_stream = session.send_audio(user_audio)
            
            # 3. Process TTS in a separate task to maintain low latency
            await stream_to_tts(text_stream, websocket)