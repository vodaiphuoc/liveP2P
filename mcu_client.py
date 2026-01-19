import asyncio, websockets, pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512
PCM_CHUNK_BYTES = CHUNK_SIZE * 2  # int16 = 2 bytes

pya = pyaudio.PyAudio()


async def listen_audio(ws_session: websockets.ClientConnection):
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

    try:
        while True:
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
            await ws_session.send(message=data, text= False)
    except asyncio.CancelledError:
        pass
    finally:
        audio_stream.close()

async def play_audio(ws_session: websockets.ClientConnection):
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
    try:
        while True:
            new_wav: bytes = await ws_session.recv(decode=False)
            pcm_buffer.extend(new_wav)

            while len(pcm_buffer) >= PCM_CHUNK_BYTES:
                chunk = pcm_buffer[:PCM_CHUNK_BYTES]
                del pcm_buffer[:PCM_CHUNK_BYTES]

                await asyncio.to_thread(stream.write, bytes(chunk))
    except asyncio.CancelledError:
        pass
    finally:
        stream.close()

async def run():
    """Main function to run the audio loop."""
    try:
        async with websockets.connect("ws://localhost:8080/voice", max_size=None) as ws:
            await asyncio.gather(
                listen_audio(ws),
                play_audio(ws)
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
