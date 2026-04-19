import time
import numpy as np
import whisper

print("Loading small...")
t0 = time.time()
model = whisper.load_model("small")
print(f"Load: {time.time()-t0:.1f}s")

# 3 seconds of low noise so transcribe has something to chew on
rng = np.random.default_rng(0)
audio = (rng.standard_normal(16000 * 3) * 0.001).astype(np.float32)

t0 = time.time()
result = model.transcribe(audio, language="en", fp16=False)
print(f"Transcribe 3s silence: {time.time()-t0:.2f}s")
print(f"Text: {result['text']!r}")
