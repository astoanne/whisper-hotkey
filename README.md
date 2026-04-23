# Whisper Hotkey Dictation

Local, GPU-accelerated push-to-talk dictation for Windows. Tap a hotkey (or twist your keyboard's volume dial), speak, release — the transcript is auto-pasted into whatever app has focus. An optional local LLM (via Ollama) can clean up grammar or apply spoken corrections. Nothing leaves your machine.

Cute cat meows for start and stop, because why not.

---

## Features

- **Push-to-talk** via `F9` or a keyboard volume dial (multimedia keys)
- **faster-whisper** transcription — `large-v3-turbo` on GPU (~0.5 s for 10 s audio on RTX 4060), `small` on CPU
- **Auto-detects language** per utterance: English, Mandarin, Cantonese (`yue`), and ~95 others
- **Organize**: tidy up grammar/punctuation on the current text field via a local LLM (Ollama + `llama3.2:3b` by default)
- **Correction**: say *"sorry I mean X"* and the LLM rewrites the field with only that change
- **Volume-key suppression** while running so the dial doesn't blast your speakers; normal volume behavior restored when the script exits
- **No network calls** — Whisper runs locally, Ollama runs locally

---

## Recommended hardware

### Fast path (GPU, what this was built on)
- Windows 10 or 11
- NVIDIA GPU with **≥ 6 GB VRAM** (RTX 3060 / 4060 / equivalent or better)
- 16 GB RAM
- Python 3.10 – 3.12
- ~5 GB free disk for Whisper weights + Ollama + `llama3.2:3b`

Performance on RTX 4060: 10 s of speech → ~0.5 s transcription, ~2 s organize.

### Minimum (CPU-only, still usable)
- Windows 10 or 11
- **8 GB RAM**
- Any modern x86-64 CPU with AVX2
- Python 3.10 – 3.12
- ~2 GB free disk (Whisper `small` + Ollama + `llama3.2:3b`)

Expected CPU speed with `small`: ~0.5× realtime (10 s audio → ~20 s to transcribe). `tiny` is ~3× faster but noticeably less accurate. If you can't spare 4 GB RAM for Ollama, skip the LLM steps and just use plain dictation.

---

## Install

### 1. Python dependencies

```cmd
pip install -r requirements.txt
```

### 2. (GPU only) CUDA runtime wheels

```cmd
pip install -r requirements-gpu.txt
```

This installs `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`. The script adds their DLL directories to the Windows loader at startup — you do **not** need a system-wide CUDA toolkit install.

You do need a recent NVIDIA GPU driver (≥ 551 works; `nvidia-smi` should print a driver version).

### 3. (Optional) Ollama — for organize / correction

Skip if you only want raw dictation.

1. Install Ollama from <https://ollama.com/download>.
2. Pull the model:

   ```cmd
   ollama pull llama3.2:3b
   ```

   (2 GB download. If you're tight on space, use `qwen2.5:1.5b` or `llama3.2:1b` and set `ORGANIZE_MODEL` accordingly — see below.)

3. Ollama runs a local server on `127.0.0.1:11434` as a background service. `ollama serve` will be running automatically after install.

**Moving the model cache off C:** On Windows, Ollama defaults to `%USERPROFILE%\.ollama\models`. To put weights on another drive, set the env var before launching the server:

```cmd
setx OLLAMA_MODELS "D:\Ollama\models"
```

Then restart the Ollama tray app / `ollama serve`. There's also a sample `start_ollama.cmd` + `start_ollama.vbs` pattern in this repo's docs for auto-launching a custom path at login — not required.

### 4. Run

```cmd
python whisper_hotkey.py
```

First run downloads the Whisper weights into `%USERPROFILE%\.cache\huggingface\`.

---

## Configuration (environment variables)

All optional. Defaults in parentheses.

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_MODEL` | `large-v3-turbo` on CUDA, `small` on CPU | Any faster-whisper model ID: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `WHISPER_COMPUTE` | `float16` on CUDA, `int8` on CPU | ctranslate2 compute type. Try `int8_float16` for lower VRAM on GPU |
| `ORGANIZE_MODEL` | `llama3.2:3b` | Any Ollama model you've pulled |
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | Ollama chat endpoint |
| `AUTO_PASTE` | `1` | `0` skips the `Ctrl+V` after dictation (copy-only) |

### CPU-only example

```cmd
set WHISPER_DEVICE=cpu
set WHISPER_MODEL=small
set WHISPER_COMPUTE=int8
python whisper_hotkey.py
```

### Smaller LLM example

```cmd
ollama pull qwen2.5:1.5b
set ORGANIZE_MODEL=qwen2.5:1.5b
python whisper_hotkey.py
```

---

## Usage

### Keyboard (`F9` / `F10`)

| Gesture | Action |
|---|---|
| Single tap `F9`, speak, tap again (or any key) | Transcribe |
| Double tap `F9`, speak, tap to stop | Transcribe + organize |
| Hold `F9` ≥ 0.45 s | Organize the current text field (no new recording) |
| `F10` | Quit |

### Volume dial (optional, if your keyboard has one)

The script captures multimedia volume keys via a low-level Windows hook. Your physical volume dial keeps its normal function **after the script exits**.

| Gesture | Action |
|---|---|
| Tiny clockwise (1–2 detents) | Redo (`Ctrl+Y`) |
| Quick clockwise (3+ detents) | Start / stop recording |
| Tiny counter-clockwise (1–2 detents) | Undo (`Ctrl+Z`) |
| Quick counter-clockwise (3+ detents) | Empty the current field (`Ctrl+A`, `Delete`) |
| Press the dial (mute key) | Send `Enter` |
| Counter-clockwise → clockwise | Correction mode — speak *"sorry I mean X"* and the LLM revises the current field |
| Clockwise → counter-clockwise | Organize the current field via the LLM |

To check what your keyboard's dial actually emits, run `probe_dial.py` and twist it. If it's silent there, your dial uses a vendor HID driver that bypasses Windows' keyboard hook and can't be captured.

---

## How correction works

1. Put your cursor in any text field that already has content.
2. Quick counter-clockwise → clockwise on the dial.
3. Speak the correction, e.g. *"sorry I mean four PM"*.
4. Twist clockwise again (or tap `F9`) to stop.
5. The script: grabs the whole field via `Ctrl+A`, `Ctrl+C` → sends it plus your spoken correction to the LLM → pastes the revised text back with `Ctrl+A`, `Ctrl+V` → restores your clipboard.

The system prompt is tuned to change **only what the correction explicitly targets** and leave every other word, punctuation mark, and line break exactly alone.

---

## Files in this repo

| File | Purpose |
|---|---|
| `whisper_hotkey.py` | Main script — load once, listen forever |
| `requirements.txt` | Core Python deps |
| `requirements-gpu.txt` | Extra wheels for CUDA acceleration |
| `sounds/start.wav`, `sounds/stop.wav` | Cat meows 🐱 |
| `probe_dial.py` | Prints the virtual-key names your volume dial emits — run this first if the dial gestures don't fire |
| `smoke_test.py` | 3-second silence → Whisper `small` sanity check |
| `bench_gpu.py` | Micro-benchmark across CUDA fp16 / int8 / CPU int8 |

---

## Troubleshooting

**`Could not load library libcublas.so.12` / `cublas64_12.dll not found`**
You skipped `requirements-gpu.txt` or `nvidia-smi` shows no GPU. Install the GPU wheels, or switch to `WHISPER_DEVICE=cpu`.

**Volume still changes when you twist the dial**
Either the ctypes hook didn't install (look for `[mediahook] SetWindowsHookExW FAILED` in stdout) or your keyboard emits volume over a vendor HID path that bypasses `WH_KEYBOARD_LL`. Confirm with `probe_dial.py` — if that sees the dial, our hook should too.

**`[organize err] connect: HTTP Error 404` or connection refused**
Ollama isn't running, is running on a different port, or hasn't pulled the model. Check `curl http://127.0.0.1:11434/api/tags`.

**Correction / organize takes forever the first time**
Ollama lazily loads model weights into VRAM on first call (can be 30–60 s for a 3 B-param model on a warm disk, longer on cold). Subsequent calls are fast while `OLLAMA_KEEP_ALIVE` keeps it resident.

**Transcription comes out in English when you spoke Chinese**
Make sure you're running a recent version of this script — earlier revisions forced `language="en"`. Current code uses `language=None` so Whisper auto-detects each utterance. The detected language is printed on every transcription, e.g. `lang=yue (0.94)`.

---

## License

MIT.
