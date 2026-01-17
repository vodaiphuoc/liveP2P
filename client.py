import asyncio, websockets, pyaudio


FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512



async def main():
    async with websockets.connect("wss://mullet-immortal-labrador.ngrok-free.app/voice", max_size=None) as ws:
        
        def callback(in_data: bytes, frame_count, time_info, status):
            
            try:
                asyncio.run(ws.send("hello"))
            except websockets.exceptions.ConnectionClosed as e:
                print("Connection closed, stopping audio stream.")
                raise e
            
            return (in_data, pyaudio.paContinue)

        pya = pyaudio.PyAudio()
        mic_info = pya.get_default_input_device_info()

        audio_stream = pya.open(
            format= FORMAT,
            channels= CHANNELS,
            rate= SEND_SAMPLE_RATE, 
            input= True, 
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=callback
            )
        
        while audio_stream.is_active():
            await asyncio.sleep(0.1)

        # Close the stream (5)
        audio_stream.close()

        # Release PortAudio system resources (6)
        pya.terminate()


asyncio.run(main())