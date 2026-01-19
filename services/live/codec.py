import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import re

class NeuCodecDecoder:
    def __init__(
        self,
        model_id: str = "neuphonic/neucodec-onnx-decoder-int8",
        max_workers: int = 2,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
    ):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="tts"
        )

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = intra_op_threads
        so.inter_op_num_threads = inter_op_threads

        onnx_path = hf_hub_download(
            repo_id=model_id,
            filename="model.onnx",
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            local_files_only=False,
            token=None
        )

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod    
    def float32_to_pcm16(audio_float)->bytes:
        """Convert float32 [-1, 1] to int16 bytes"""
        audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _decode_sync(self, codes: str) -> bytes:
        """
        latent: float32 [1, T, D]
        return: PCM float32 waveform
        """
        speech_ids = [int(num) for num in re.findall(r"<\|speech_(\d+)\|>", codes)]
        codes = np.array(speech_ids, dtype=np.int32)[np.newaxis, np.newaxis, :]

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: codes},
        )[0].astype(np.float32)
        
        return NeuCodecDecoder.float32_to_pcm16(outputs[0, 0, :])

    async def decode(self, codes: str) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._decode_sync,
            codes,
        )

    def close(self):
        self.executor.shutdown(wait=True)
