import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pyaudio
import websockets
load_dotenv()

TTS_URL = os.environ['DEPLOY_DOMAIN']

client = genai.Client(api_key=os.environ['GEMINI_API'], http_options={"api_version": "v1alpha"})
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"


SYSTEM_INST = """
Bạn là một voicebot thân thiện với con người, hãy luôn trả lời câu hỏi một
cách vui vẻ, và bằng tiếng Việt (Vietnamese)
"""

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    output_audio_transcription = {},
    enable_affective_dialog = True,
    system_instruction = SYSTEM_INST
)

pya = pyaudio.PyAudio()
# --- pyaudio config ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

audio_queue_mic = asyncio.Queue(maxsize=5)
audio_queue_output = asyncio.Queue(maxsize=20)

PCM_CHUNK_BYTES = CHUNK_SIZE * 2  # int16 = 2 bytes


async def listen_audio():
    """Listens for audio and puts it into the mic audio queue."""
    global audio_stream
    mic_info = pya.get_default_input_device_info()
    audio_stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_SAMPLE_RATE,
        input=True,
        input_device_index=mic_info["index"],
        frames_per_buffer=CHUNK_SIZE,
    )
    kwargs = {"exception_on_overflow": False} if __debug__ else {}
    while True:
        data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
        await audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})

async def send_realtime(live_session):
    """Sends audio from the mic audio queue to the GenAI session."""
    while True:
        msg = await audio_queue_mic.get()
        await live_session.send_realtime_input(audio=msg)


async def receive_live_response(live_session, tts_ws):
    """Receives responses from GenAI and puts audio data into the speaker audio queue."""
    while True:
        turn = live_session.receive()
        async for response in turn:
            if response.server_content.output_transcription:
                response_segment_text = response.server_content.output_transcription.text
                print('receive response_segment_text: ',response_segment_text)
                await tts_ws.send(response_segment_text)

async def receive_audio_response(tts_ws):
    """Receives responses from TTS service"""
    while True:
        audio_response = await tts_ws.recv()
        print('receive audio response: ', type(audio_response))
        audio_queue_output.put_nowait(audio_response)

async def play_audio():
    """Plays audio from the speaker audio queue."""
    speaker_info = pya.get_default_output_device_info()
    pcm_buffer = bytearray()
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,
        output=True,
        output_device_index = speaker_info['index'],
        frames_per_buffer=CHUNK_SIZE,
    )
    while True:
        data = await audio_queue_output.get()
        pcm_buffer.extend(data)

        while len(pcm_buffer) >= PCM_CHUNK_BYTES:
            chunk = pcm_buffer[:PCM_CHUNK_BYTES]
            del pcm_buffer[:PCM_CHUNK_BYTES]

            await asyncio.to_thread(stream.write, bytes(chunk))

async def run():
    """Main function to run the audio loop."""
    try:
        async with client.aio.live.connect(
            model=MODEL, config=CONFIG
        ) as live_session, websockets.connect(f"wss://{TTS_URL}/voice", max_size=None) as tts_ws:
            
            await asyncio.gather(
                listen_audio(),
                send_realtime(live_session),
                receive_live_response(live_session, tts_ws),
                receive_audio_response(tts_ws),
                play_audio()
            )
            
    except asyncio.CancelledError:
        pass
    finally:
        pya.terminate()
        print("\nConnection closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Interrupted by user.")
