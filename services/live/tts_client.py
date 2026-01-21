from utils.phonemize_text import phonemize_with_dict
import anyio
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
import aiohttp

from codec import NeuCodecDecoder

async def tts_worker(
        receive_live_output_stream: MemoryObjectReceiveStream[str],
        send_tts_stream: MemoryObjectSendStream[bytes],
        tts_session: aiohttp.ClientSession,
        codec_decoder: NeuCodecDecoder,
        ref_codes: list[int]|None = None, 
        ref_text: str = None,
        temperature: float = 1.0, 
        top_k: int = 50
    ):
    r"""
    A worker for processing chunks and send requests to TTS server
    """
    limiter = anyio.CapacityLimiter(2)

    async def process_chunk(
            chunk: str, 
            ref_codes: list[int]|None = None, 
            ref_text: str = None
            ):
        async with limiter:
            input_text = phonemize_with_dict(chunk)

            if ref_text:
                ref_text = phonemize_with_dict(ref_text)
                codes_str = "".join([f"<|speech_{idx}|>" for idx in ref_codes])
                prompt = (
                    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text} {input_text}"
                    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"
                )
            else:
                prompt = (
                    f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{input_text}"
                    f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
                )
            
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "top_k": top_k,
                "stop": ["<|SPEECH_GENERATION_END|>"],
            }

            async with tts_session.post("v1/completions", json=payload) as resp:
                data = await resp.json()

                codes:str = data["choices"][0]["text"]
                wav_array = await codec_decoder.decode(codes)
                await send_tts_stream.send(wav_array)

    async with anyio.create_task_group() as tg:
        async for chunk in receive_live_output_stream:
            tg.start_soon(process_chunk, chunk, ref_codes, ref_text)

