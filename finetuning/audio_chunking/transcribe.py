from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    "long.wav",
    language="vi",
    vad_filter=True
)

with open("long_segments.json", "w", encoding="utf-8") as f:
    for s in segments:
        f.write(
            f"{s.start:.2f}\t{s.end:.2f}\t{s.text.strip()}\n"
        )
