import soundfile as sf
import os

os.makedirs("rough_chunks/wav", exist_ok=True)
os.makedirs("rough_chunks/txt", exist_ok=True)

audio, sr = sf.read("long.wav")

def sec_to_idx(t): return int(t * sr)

with open("long_segments.json", encoding="utf-8") as f:
    for i, line in enumerate(f):
        start, end, text = line.strip().split("\t")
        start, end = float(start), float(end)

        if end - start < 1.0:
            continue

        # padding
        s = max(0, start - 0.1)
        e = min(len(audio)/sr, end + 0.2)

        chunk = audio[sec_to_idx(s):sec_to_idx(e)]

        wav_path = f"rough_chunks/wav/chunk_{i:05d}.wav"
        txt_path = f"rough_chunks/txt/chunk_{i:05d}.txt"

        sf.write(wav_path, chunk, sr)

        with open(txt_path, "w", encoding="utf-8") as t:
            t.write(text)
