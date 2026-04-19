import time
import numpy as np
from faster_whisper import WhisperModel

# 30s of soft noise (not real speech, but exercises the full pipeline)
rng = np.random.default_rng(0)
audio = (rng.standard_normal(16000 * 30) * 0.001).astype(np.float32)

for device, compute in [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]:
    print(f"\n=== {device} / {compute} ===")
    t0 = time.time()
    model = WhisperModel("small", device=device, compute_type=compute)
    print(f"load: {time.time()-t0:.2f}s")

    # warmup
    list(model.transcribe(audio[:16000], language="en")[0])

    t0 = time.time()
    segments, info = model.transcribe(audio, language="en", beam_size=1)
    segments = list(segments)
    dt = time.time() - t0
    print(f"transcribe 30s: {dt:.2f}s  ({30/dt:.1f}x realtime)")
    del model
