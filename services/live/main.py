import re
import aiohttp
import anyio
import numpy as np

from tts_client import tts_worker
from llm import LLMConfig
from codec import NeuCodecDecoder


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from dotenv import load_dotenv
from loguru import logger
from contextlib import asynccontextmanager
import os

load_dotenv()

APPLICATION_PORT = os.environ['APPLICATION_PORT']
HTTPS_SERVER = os.environ['HTTPS_SERVER']

TTS_PORT = os.environ['TTS_PORT']
TTS_HOST = os.environ['TTS_HOST']

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up Ngrok Tunnel")

    app.state.llm_config = LLMConfig()
    app.state.codec = NeuCodecDecoder()

    yield
    logger.info("Tearing Down Ngrok Tunnel")
    app.state.codec.close()

app = FastAPI(lifespan=lifespan)


@app.get("/healthcheck")
async def healthcheck():
    return JSONResponse(content="oke")

@app.websocket("/voice")
async def voice(websocket: WebSocket):
    await websocket.accept()
    
    llm_config: LLMConfig = websocket.app.state.llm_config
    send_live_output_stream, receive_live_output_stream = anyio.create_memory_object_stream[str](max_buffer_size=8)
    send_tts_stream, receive_tts_stream = anyio.create_memory_object_stream[bytes](max_buffer_size=8)
    try:
        async with llm_config.client.aio.live.connect(
            model=llm_config.model_id,
            config=llm_config.live_config
        ) as live_session, \
            aiohttp.ClientSession(
                base_url=f"http://{TTS_HOST}:{TTS_PORT}/"
        ) as tts_session:

            async def _send_audio():
                r"""
                Send audio bytes to live api
                """
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await live_session.send_realtime_input(
                            audio={"data": data, "mime_type": "audio/pcm"}
                        )
                except WebSocketDisconnect as e:
                    await send_live_output_stream.aclose()
                    await send_tts_stream.aclose()
                    raise e

            async def _receive_responses():
                r"""
                Receive audio bytes to live api, produce meaningful chunks, 
                then put chunks into 
                """
                response_text:str = ""
                async for response in live_session.receive():
                    if response.server_content.output_transcription:
                        response_text += response.server_content.output_transcription.text
                    
                    span_search = re.search(r'[.!?]\s', response_text)

                    if span_search:
                        end_index = span_search.span()[-1]
                        to_speak_text = response_text[:end_index]
                        
                        print('receive response_segment_text: ',to_speak_text)
                        await send_live_output_stream.send(to_speak_text)

                        # trimming
                        response_text = response_text[end_index:]
            
            async def _send_audio_response():
                async for wav in receive_tts_stream:
                    if isinstance(wav, bytes):
                        await websocket.send_bytes(wav)

            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    _send_audio, 
                    name="send audio to live api"
                )
                tg.start_soon(
                    _receive_responses, 
                    name = "get response from live api"
                )
                tg.start_soon(
                    tts_worker,
                    receive_live_output_stream, 
                    send_tts_stream,
                    tts_session, 
                    websocket.app.state.codec, 
                    name = "tts client worker"
                )
                tg.start_soon(
                    _send_audio_response,
                    name="send audio response through websocket"
                )

    except WebSocketDisconnect as e:
        print('disconnected: ',e)


async def main_run():
    config = uvicorn.Config(
        "main:app", 
    	host=HTTPS_SERVER,
        port=int(APPLICATION_PORT),
    	reload=True,
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main_run())