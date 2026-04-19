import os
import site
import pathlib
import sys
import time
import threading
import json
import urllib.request
import urllib.error
import ctypes
from ctypes import wintypes
import atexit

# Make CUDA DLLs (cublas, cudnn, nvrtc) discoverable for ctranslate2 on Windows.
for _sp in site.getsitepackages() + [site.getusersitepackages()]:
    for _sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
        _d = pathlib.Path(_sp) / _sub
        if _d.exists():
            os.add_dll_directory(str(_d))

import winsound
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
from faster_whisper import WhisperModel, BatchedInferencePipeline

HOTKEY = "f9"
QUIT_KEY = "f10"
MULTI_TAP_WINDOW = 0.28
LONG_TAP_THRESHOLD = 0.45  # hold >= this -> "long tap"
SAMPLE_RATE = 16000
AUTO_PASTE = True

# All tunables below are overridable via environment variables so the same
# script runs on a GPU box (large-v3-turbo / float16) or a CPU-only machine
# (small / int8).
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda").lower()
MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo" if DEVICE == "cuda" else "small")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16" if DEVICE == "cuda" else "int8")

SOUND_DIR = pathlib.Path(__file__).parent / "sounds"
SOUND_START = str(SOUND_DIR / "start.wav")
SOUND_STOP = str(SOUND_DIR / "stop.wav")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
ORGANIZE_MODEL = os.environ.get("ORGANIZE_MODEL", "llama3.2:3b")
ORGANIZE_SYSTEM = (
    "You fix mistakes in dictated text. Make the MINIMUM changes necessary: "
    "fix obvious transcription errors, misspellings, wrong-word homophones, "
    "and clear grammar or punctuation mistakes. Do NOT rephrase, reword, "
    "reorganize, restructure sentences, substitute synonyms, change tone, "
    "'improve flow', or tidy up style. If a sentence is already grammatical "
    "and understandable, leave every word and punctuation mark exactly as "
    "written. Preserve the user's voice, phrasing, contractions, and informal "
    "language. Output ONLY the text with no preamble, no commentary, no quotes."
)
CORRECTION_SYSTEM = (
    "You apply a spoken correction to existing text. You receive the full "
    "ORIGINAL text and a CORRECTION phrase (e.g. 'sorry I mean X'). "
    "Apply ONLY what the correction EXPLICITLY asks to change. "
    "Do NOT rephrase, reword, reorganize, fix grammar, fix spelling, fix "
    "punctuation, substitute synonyms, or adjust anything the correction does "
    "not directly target. Every other word, punctuation mark, line break, "
    "capitalization, and formatting element must be preserved EXACTLY as in "
    "the ORIGINAL. If the correction is ambiguous, make the smallest possible "
    "edit that satisfies it. Output ONLY the revised text with no preamble, "
    "no commentary, no quotes."
)


def play(path: str):
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[sound err] {e}")


print(f"loading {MODEL_NAME} on {DEVICE}/{COMPUTE}...")
_t = time.time()
models = {
    MODEL_NAME: BatchedInferencePipeline(
        model=WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)
    ),
}
print(f"ready ({time.time()-_t:.1f}s)")
print(f"{HOTKEY}: 1x=transcribe  2x=transcribe+organize  hold={LONG_TAP_THRESHOLD:.1f}s=organize current field")
print(f"quit:    {QUIT_KEY}\n")


# tap count -> (model, do_organize_after)
TAP_ACTIONS = {
    1: (MODEL_NAME, False),
    2: (MODEL_NAME, True),
}

state = {
    "recording": False,
    "stream": None,
    "chunks": [],
    "start_time": 0.0,
    "model_name": None,
    "organize_after": False,
    "correction_of": False,  # bool -- if True, transcribe then revise current field
    "last_dictation": None,  # last text we pasted, for correction gesture
    "tap_count": 0,
    "tap_timer": None,
    "key_down": False,
    "press_time": 0.0,
}
lock = threading.Lock()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    state["chunks"].append(indata.copy())


def _ollama_chat(system: str, user: str, tag: str = "llm") -> str | None:
    payload = json.dumps({
        "model": ORGANIZE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip() or None
    except urllib.error.URLError as e:
        print(f"[{tag} err] connect: {e}. Is ollama running on 11434?")
        return None
    except Exception as e:
        print(f"[{tag} err] {e}")
        return None


def call_organize(text: str) -> str | None:
    return _ollama_chat(ORGANIZE_SYSTEM, text, tag="organize")


def call_correction(original: str, correction: str) -> str | None:
    user_msg = f"ORIGINAL: {original}\n\nCORRECTION: {correction}"
    return _ollama_chat(CORRECTION_SYSTEM, user_msg, tag="correction")


def start_recording(model_name: str, organize_after: bool, correction_of: bool = False):
    state["chunks"] = []
    state["start_time"] = time.time()
    state["model_name"] = model_name
    state["organize_after"] = organize_after
    state["correction_of"] = correction_of
    state["stream"] = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    )
    state["stream"].start()
    state["recording"] = True
    play(SOUND_START)
    if correction_of:
        print("[REC] correction mode -- will revise whole field")
    else:
        suffix = " +organize" if organize_after else ""
        print(f"[REC] {model_name}{suffix} -- tap F9 to stop")


def stop_and_transcribe():
    state["stream"].stop()
    state["stream"].close()
    state["stream"] = None
    state["recording"] = False
    play(SOUND_STOP)
    dur = time.time() - state["start_time"]

    if not state["chunks"]:
        print("[--] no audio")
        return

    audio = np.concatenate(state["chunks"], axis=0).flatten().astype(np.float32)
    model = models[state["model_name"]]

    t0 = time.time()
    segments, info = model.transcribe(audio, language=None, beam_size=1, batch_size=8)
    text = " ".join(s.text.strip() for s in segments).strip()
    elapsed = time.time() - t0

    lang = getattr(info, "language", "?")
    lang_prob = getattr(info, "language_probability", 0.0)

    if not text:
        print(f"[--] empty transcript ({dur:.1f}s audio, {elapsed:.2f}s infer, lang={lang})")
        return

    print(f"[OK] {dur:.1f}s audio, {elapsed:.2f}s infer, lang={lang} ({lang_prob:.2f}) -> {text}")

    # correction mode: grab whole field, LLM revises using spoken correction, paste back
    if state["correction_of"]:
        state["correction_of"] = False
        saved_clip = pyperclip.paste()
        keyboard.send("ctrl+a")
        time.sleep(0.05)
        keyboard.send("ctrl+c")
        time.sleep(0.18)
        original = pyperclip.paste()
        if not original.strip():
            print("[COR] field is empty -- pasting raw transcript instead")
            pyperclip.copy(text)
            time.sleep(0.02)
            keyboard.send("ctrl+v")
            time.sleep(0.1)
            pyperclip.copy(saved_clip)
            state["last_dictation"] = text
            return
        print(f"[COR] revising {len(original)} chars via LLM (correction: {text!r})...")
        t0 = time.time()
        revised = call_correction(original, text)
        if not revised:
            print("[COR] failed -- field unchanged")
            pyperclip.copy(saved_clip)
            return
        print(f"[COR] {time.time()-t0:.1f}s -> {len(revised)} chars")
        pyperclip.copy(revised)
        keyboard.send("ctrl+a")
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(0.1)
        pyperclip.copy(saved_clip)
        state["last_dictation"] = revised
        return

    if state["organize_after"]:
        print("[ORG] organizing...")
        t0 = time.time()
        cleaned = call_organize(text)
        if cleaned:
            text = cleaned
            print(f"[ORG] {time.time()-t0:.1f}s -> {text}")
        else:
            print("[ORG] failed, using raw transcript")

    pyperclip.copy(text)
    if AUTO_PASTE:
        time.sleep(0.05)
        keyboard.send("ctrl+v")
    state["last_dictation"] = text


def organize_current_field():
    saved_clip = pyperclip.paste()
    keyboard.send("ctrl+a")
    time.sleep(0.05)
    keyboard.send("ctrl+c")
    time.sleep(0.18)
    original = pyperclip.paste()

    if not original.strip():
        print("[ORG] nothing in field to organize")
        pyperclip.copy(saved_clip)
        return

    print(f"[ORG] organizing {len(original)} chars of current field...")
    t0 = time.time()
    cleaned = call_organize(original)
    if not cleaned:
        print("[ORG] failed, field unchanged")
        pyperclip.copy(saved_clip)
        return

    pyperclip.copy(cleaned)
    keyboard.send("ctrl+a")
    time.sleep(0.05)
    keyboard.send("ctrl+v")
    time.sleep(0.1)
    pyperclip.copy(saved_clip)  # restore original clipboard
    print(f"[ORG] done in {time.time()-t0:.1f}s -> {len(cleaned)} chars")


def dispatch_tap(tap_count: int):
    with lock:
        try:
            action = TAP_ACTIONS.get(tap_count)
            if action is None:
                print(f"[??] {tap_count} taps -- ignored")
                return
            model_name, organize_after = action
            start_recording(model_name, organize_after)
        except Exception as e:
            print(f"[ERR] {e}")
            state["recording"] = False


def _on_press(e):
    if state["key_down"]:
        return  # suppress OS key-repeat
    state["key_down"] = True
    state["press_time"] = time.time()


def _on_release(e):
    if not state["key_down"]:
        return
    state["key_down"] = False
    hold = time.time() - state["press_time"]

    # While recording, any release of the hotkey stops immediately.
    if state["recording"]:
        threading.Thread(target=lambda: (lock.acquire(), stop_and_transcribe(), lock.release()), daemon=True).start()
        return

    if hold >= LONG_TAP_THRESHOLD:
        # long tap = organize current field; cancel any pending multi-tap
        state["tap_count"] = 0
        if state["tap_timer"] is not None:
            state["tap_timer"].cancel()
            state["tap_timer"] = None
        threading.Thread(target=organize_current_field, daemon=True).start()
        return

    # short tap -> accumulate for multi-tap
    state["tap_count"] += 1
    if state["tap_timer"] is not None:
        state["tap_timer"].cancel()

    def fire():
        count = state["tap_count"]
        state["tap_count"] = 0
        state["tap_timer"] = None
        threading.Thread(target=dispatch_tap, args=(count,), daemon=True).start()

    state["tap_timer"] = threading.Timer(MULTI_TAP_WINDOW, fire)
    state["tap_timer"].start()


keyboard.on_press_key(HOTKEY, _on_press, suppress=False)
keyboard.on_release_key(HOTKEY, _on_release, suppress=False)


# ---- volume-dial gestures (keyboard media keys) --------------------------
DIAL_RELEASE_WINDOW = 0.45  # wait this long after last detent before firing
DIAL_TINY_MAX = 2           # 1-2 detents = "tiny", 3+ = "quick"

dial_state = {
    "cw": 0,
    "ccw": 0,
    "first_dir": None,   # "cw" or "ccw" -- direction of first detent in gesture
    "timer": None,
    "last_mute_time": 0.0,
}


def _start_toggle_record():
    with lock:
        if state["recording"]:
            stop_and_transcribe()
        else:
            start_recording(MODEL_NAME, False)


def _start_correction_recording():
    with lock:
        if state["recording"]:
            return
        start_recording(MODEL_NAME, False, correction_of=True)


def _empty_field():
    print("[dial] quick CCW -> empty field (ctrl+a, delete)")
    keyboard.send("ctrl+a")
    time.sleep(0.04)
    keyboard.send("delete")


def _fire_dial():
    cw, ccw = dial_state["cw"], dial_state["ccw"]
    first = dial_state["first_dir"]
    dial_state["cw"] = 0
    dial_state["ccw"] = 0
    dial_state["first_dir"] = None
    dial_state["timer"] = None

    if state["recording"]:
        print("[dial] rotation during recording -> stop")
        threading.Thread(target=_start_toggle_record, daemon=True).start()
        return

    # two-direction gesture: both sides saw >= 2 detents
    if cw >= 2 and ccw >= 2:
        if first == "ccw":
            print(f"[dial] CCW->CW ({ccw}+{cw}) -> correction mode")
            threading.Thread(target=_start_correction_recording, daemon=True).start()
        else:  # first == "cw"
            print(f"[dial] CW->CCW ({cw}+{ccw}) -> organize current field")
            threading.Thread(target=organize_current_field, daemon=True).start()
        return

    if cw > 0 and ccw > 0:
        print(f"[dial] mixed too small cw={cw} ccw={ccw} -- ignored")
        return

    if cw > 0:
        if cw <= DIAL_TINY_MAX:
            print(f"[dial] tiny CW ({cw}) -> redo")
            keyboard.send("ctrl+y")
        else:
            print(f"[dial] quick CW ({cw}) -> start recording")
            threading.Thread(target=_start_toggle_record, daemon=True).start()
    else:
        if ccw <= DIAL_TINY_MAX:
            print(f"[dial] tiny CCW ({ccw}) -> undo")
            keyboard.send("ctrl+z")
        else:
            _empty_field()


def _reset_dial_timer():
    if dial_state["timer"] is not None:
        dial_state["timer"].cancel()
    dial_state["timer"] = threading.Timer(DIAL_RELEASE_WINDOW, _fire_dial)
    dial_state["timer"].start()


def _on_media_key(vk_name: str):
    # Called from the ctypes hook thread -- must return quickly.
    if vk_name == "volume up":
        if dial_state["first_dir"] is None:
            dial_state["first_dir"] = "cw"
        dial_state["cw"] += 1
        _reset_dial_timer()
    elif vk_name == "volume down":
        if dial_state["first_dir"] is None:
            dial_state["first_dir"] = "ccw"
        dial_state["ccw"] += 1
        _reset_dial_timer()
    elif vk_name == "volume mute":
        now = time.time()
        if now - dial_state["last_mute_time"] < 0.25:
            return
        dial_state["last_mute_time"] = now
        print("[dial] press -> send (enter)")
        keyboard.send("enter")


# ---- ctypes WH_KEYBOARD_LL hook that suppresses volume keys -------------
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_WH_KEYBOARD_LL = 13
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_SYSKEYDOWN = 0x0104
_WM_SYSKEYUP = 0x0105
_WM_QUIT = 0x0012
_VK = {0xAF: "volume up", 0xAE: "volume down", 0xAD: "volume mute"}


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


_HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
)

_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD
]
_user32.SetWindowsHookExW.restype = wintypes.HHOOK
_user32.CallNextHookEx.argtypes = [
    wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
]
_user32.CallNextHookEx.restype = ctypes.c_long
_user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL
_user32.GetMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT
]
_user32.GetMessageW.restype = ctypes.c_int
_user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
]
_user32.PostThreadMessageW.restype = wintypes.BOOL
_kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetCurrentThreadId.argtypes = []
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD


class MediaKeyHook:
    def __init__(self):
        self.suppress = True
        self._hook_id = None
        self._thread_id = None
        self._proc_ref = None  # keep the WINFUNCTYPE alive
        self._ready = threading.Event()
        self._install_err = None
        self._seen_media = False

    def _run(self):
        def _proc(nCode, wParam, lParam):
            if nCode == 0:
                kb = ctypes.cast(lParam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                name = _VK.get(kb.vkCode)
                if name is not None:
                    if not self._seen_media:
                        self._seen_media = True
                        print(f"[mediahook] first media event seen: vk=0x{kb.vkCode:02X} name={name!r} wParam=0x{wParam:04X}")
                    if wParam in (_WM_KEYDOWN, _WM_SYSKEYDOWN):
                        try:
                            _on_media_key(name)
                        except Exception as e:
                            print(f"[mediahook err] {e}")
                    if self.suppress:
                        return 1
            return _user32.CallNextHookEx(self._hook_id, nCode, wParam, lParam)

        self._proc_ref = _HOOKPROC(_proc)
        self._thread_id = _kernel32.GetCurrentThreadId()
        # WH_KEYBOARD_LL accepts NULL hMod.
        self._hook_id = _user32.SetWindowsHookExW(
            _WH_KEYBOARD_LL, self._proc_ref, None, 0
        )
        if not self._hook_id:
            self._install_err = ctypes.get_last_error()
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        while _user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="mediahook").start()
        if not self._ready.wait(timeout=2.0):
            print("[mediahook] hook thread did not signal ready in 2s")
            return
        if self._hook_id:
            print(f"[mediahook] installed OK hook_id={self._hook_id} tid={self._thread_id}")
        else:
            print(f"[mediahook] SetWindowsHookExW FAILED err={self._install_err}")

    def stop(self):
        if self._hook_id:
            try:
                _user32.UnhookWindowsHookEx(self._hook_id)
            except Exception:
                pass
            self._hook_id = None
        if self._thread_id:
            _user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
            self._thread_id = None


_media_hook = MediaKeyHook()
_media_hook.start()
atexit.register(_media_hook.stop)

print("dial:   tinyCW=redo  quickCW=record  tinyCCW=undo  quickCCW=empty")
print("        CCW->CW=correction  CW->CCW=organize  press=send  (F9/F10 still active)")
print("volume: media keys suppressed while running; normal behavior restored on exit")


try:
    keyboard.wait(QUIT_KEY)
except KeyboardInterrupt:
    pass
print("bye")
