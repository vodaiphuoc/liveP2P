from google import genai
from google.genai import types
import os

class LLMConfig(object):

    SYSTEM_INST = """
        Bạn tên là PHước
        Bạn là một voicebot thân thiện với con người, hãy luôn trả lời câu hỏi một
        cách vui vẻ, và bằng tiếng Việt (Vietnamese). Có thể dùng các ký tự đặc biệt 
        như . , !, ; để thể hiện ngắt quảng những ý muốn nói.
        """

    def __init__(self):
        self.client = genai.Client(api_key=os.environ['GEMINI_API'], http_options={"api_version": "v1alpha"})
        self.model_id = "gemini-2.5-flash-native-audio-preview-12-2025"



        self.live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription = {},
            enable_affective_dialog = True,
            system_instruction = LLMConfig.SYSTEM_INST
        )

