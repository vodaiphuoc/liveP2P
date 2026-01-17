from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from dotenv import load_dotenv
from loguru import logger
from contextlib import asynccontextmanager
import ngrok
import os
from vieneu import Vieneu
import numpy as np

load_dotenv() 
NGROK_AUTH_TOKEN = os.environ['NGROK_AUTH_TOKEN']
APPLICATION_PORT = os.environ['APPLICATION_PORT']
HTTPS_SERVER = os.environ['HTTPS_SERVER']
DEPLOY_DOMAIN = os.environ['DEPLOY_DOMAIN']


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up Ngrok Tunnel")

    # take default params
    app.state.tts = Vieneu()
    # app.state.tts = Vieneu(
    #     mode='standard', 
    #     backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-ngoc-huyen-gguf-Q4_0",
    #     backbone_device="cpu", 
    #     codec_repo="neuphonic/neucodec-onnx-decoder-int8", 
    #     codec_device="cpu"
    # )

    app.state.voice_data = app.state.tts.get_preset_voice('Ngoc')

    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    ngrok.forward(addr = HTTPS_SERVER+':'+str(APPLICATION_PORT),
                  proto = "http",
                  domain = DEPLOY_DOMAIN
                  )
    
    yield
    logger.info("Tearing Down Ngrok Tunnel")
    ngrok.disconnect()

app = FastAPI(lifespan=lifespan)


@app.get("/healthcheck")
async def healthcheck():
    return JSONResponse(content="oke")

def float32_to_pcm16(audio_float):
    """Convert float32 [-1, 1] to int16 bytes"""
    audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()

@app.websocket("/voice")
async def voice(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            print("receive text in tts server: ", data)
            audio_outputs = websocket.app.state.tts.infer(data, voice=app.state.voice_data)
            await websocket.send_bytes(float32_to_pcm16(audio_outputs))

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