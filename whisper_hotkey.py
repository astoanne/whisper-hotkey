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
ALT_HOTKEY = "f8"  # 1 tap = correction, 2 tap = organize (for macropad key 2)
QUIT_KEY = "f10"
# COIDEA macropad dial emits these (set in the COIDEA GUI). F13-F15 are
# unused on normal keyboards so they never collide with anything else.
COIDEA_DIAL_CW = "f13"
COIDEA_DIAL_CCW = "f14"
COIDEA_DIAL_PRESS = "f15"
MULTI_TAP_WINDOW = 0.28
LONG_TAP_THRESHOLD = 0.45  # hold >= this -> "long tap"
SAMPLE_RATE = 16000
# Set AUTO_PASTE=0 to keep the PC clipboard as the only hand-off (useful when
# piping clipboard to a phone via WeChat IME / Universal Clipboard / etc).
AUTO_PASTE = os.environ.get("AUTO_PASTE", "1") not in ("0", "false", "False", "no")

# USE_MEDIA_DIAL=1 re-enables the legacy gesture handler that turns the main
# keyboard's volume dial (media keys) into record / undo / redo gestures. It
# also suppresses volume up/down/mute so the dial doesn't blast speakers.
# Leave off when you've moved to the COIDEA macropad's dial instead.
USE_MEDIA_DIAL = os.environ.get("USE_MEDIA_DIAL", "0") not in ("0", "false", "False", "no")

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
    "You organize raw dictated text into clean written form. "
    "Be ACTIVE on these: "
    "(1) fix all grammar mistakes; "
    "(2) add proper punctuation -- periods, commas, question marks, quotation "
    "marks -- and fix capitalization; "
    "(3) fix misspellings and wrong-word homophones (there/their, its/it's, "
    "etc.); "
    "(4) when the content is a sequence of items, steps, reasons, or distinct "
    "points, reformat them as a numbered list (1. ..., 2. ..., 3. ...); "
    "(5) break into paragraphs where it clearly aids readability. "
    "Do NOT: substitute synonyms, paraphrase, change tone or voice, add "
    "information not in the original, or remove content the user meant to "
    "keep. Preserve the user's wording and meaning. "
    "Output ONLY the organized text with no preamble, no commentary, no quotes."
)
CORRECTION_SYSTEM = (
    "You edit text based on a spoken correction.\n\n"
    "INPUTS:\n"
    "- ORIGINAL: the full current text.\n"
    "- CORRECTION: a short spoken instruction describing what to change.\n\n"
    "STEPS:\n"
    "1. Figure out what in ORIGINAL the correction targets. Handle these "
    "common patterns:\n"
    "   - 'sorry I mean X' / 'I mean X' / 'actually X' / 'no, X' "
    "-> the user is correcting the MOST RECENTLY mentioned item; replace "
    "that item (not the whole text) with X.\n"
    "   - 'change X to Y' / 'replace X with Y' -> substitute X with Y.\n"
    "   - 'add X' / 'also X' -> insert X at the natural location "
    "(usually the end, or next to related content).\n"
    "   - 'remove X' / 'delete the part about X' -> delete that portion.\n"
    "   - 'X should be Y' -> replace the reference to X with Y.\n"
    "2. Apply the minimal edit that satisfies the correction.\n"
    "3. Leave everything else EXACTLY as in ORIGINAL -- same words, word "
    "order, punctuation, capitalization, line breaks, formatting.\n\n"
    "HARD RULES:\n"
    "- NEVER include the CORRECTION phrase itself in the output (no 'sorry I "
    "mean', no 'actually', etc.).\n"
    "- NEVER keep both the old and new values side by side; the correction "
    "REPLACES, not appends.\n"
    "- If the target appears multiple times, prefer the LAST occurrence.\n"
    "- If you genuinely cannot identify a target, make the smallest plausible "
    "edit rather than rewriting.\n"
    "- Do not fix unrelated grammar, spelling, or punctuation.\n"
    "- If ORIGINAL and CORRECTION are in different languages, keep ORIGINAL's "
    "language.\n\n"
    "EXAMPLES:\n"
    "ORIGINAL: I'll meet him at 3pm tomorrow at the cafe.\n"
    "CORRECTION: sorry I mean 4pm\n"
    "OUTPUT: I'll meet him at 4pm tomorrow at the cafe.\n\n"
    "ORIGINAL: The top languages are Python, Go, and Rust.\n"
    "CORRECTION: replace Go with Java\n"
    "OUTPUT: The top languages are Python, Java, and Rust.\n\n"
    "ORIGINAL: Buy milk, eggs, and bread.\n"
    "CORRECTION: also butter\n"
    "OUTPUT: Buy milk, eggs, bread, and butter.\n\n"
    "Output ONLY the revised text. No preamble, no commentary, no quotes, "
    "no explanation."
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
print(f"{ALT_HOTKEY}: 1x=correction  2x=organize current field")
print(f"coidea dial ({COIDEA_DIAL_PRESS}/{COIDEA_DIAL_CW}/{COIDEA_DIAL_CCW}): press=WT tab-switch toggle  CW=redo  CCW=undo (terminal-aware)")
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
    "alt_tap_count": 0,
    "alt_tap_timer": None,
    "alt_key_down": False,
    "tab_switch_mode": False,  # COIDEA dial: True = rotate switches WT tabs
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


# ---- alt hotkey (macropad key 2): 1 tap = correction, 2 tap = organize ---
def _alt_dispatch(tap_count: int):
    with lock:
        try:
            if tap_count == 1:
                print("[alt] 1 tap -> correction mode")
                start_recording(MODEL_NAME, False, correction_of=True)
            elif tap_count == 2:
                print("[alt] 2 tap -> organize current field")
                threading.Thread(target=organize_current_field, daemon=True).start()
            else:
                print(f"[alt] {tap_count} taps -- ignored")
        except Exception as e:
            print(f"[alt ERR] {e}")
            state["recording"] = False


def _on_alt_tap():
    # Fires once per macropad Key 2 press (the chord is emitted atomically).
    # If already recording, stop it.
    if state["recording"]:
        threading.Thread(
            target=lambda: (lock.acquire(), stop_and_transcribe(), lock.release()),
            daemon=True,
        ).start()
        return

    state["alt_tap_count"] += 1
    if state["alt_tap_timer"] is not None:
        state["alt_tap_timer"].cancel()

    def fire():
        count = state["alt_tap_count"]
        state["alt_tap_count"] = 0
        state["alt_tap_timer"] = None
        threading.Thread(target=_alt_dispatch, args=(count,), daemon=True).start()

    state["alt_tap_timer"] = threading.Timer(MULTI_TAP_WINDOW, fire)
    state["alt_tap_timer"].start()


keyboard.add_hotkey(ALT_HOTKEY, _on_alt_tap, suppress=True, trigger_on_release=False)


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
    fam = _shell_family()
    if fam == "psreadline":
        print("[dial] quick CCW -> revert line (esc, psreadline)")
        keyboard.send("esc")
        return
    if fam == "bash":
        print("[dial] quick CCW -> clear line (ctrl+a, ctrl+k, bash)")
        keyboard.send("ctrl+a")
        time.sleep(0.04)
        keyboard.send("ctrl+k")
        return
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
            fam = _shell_family()
            if fam == "bash":
                print(f"[dial] tiny CW ({cw}) -> redo SKIPPED (bash has no redo)")
            elif fam == "psreadline":
                print(f"[dial] tiny CW ({cw}) -> redo SKIPPED (terminal, avoid beep)")
            else:
                print(f"[dial] tiny CW ({cw}) -> redo (ctrl+y)")
                keyboard.send("ctrl+y")
        else:
            print(f"[dial] quick CW ({cw}) -> start recording")
            threading.Thread(target=_start_toggle_record, daemon=True).start()
    else:
        if ccw <= DIAL_TINY_MAX:
            fam = _shell_family()
            if fam == "psreadline":
                # PSReadLine's Ctrl+Z is per-character. Use delete-word-back
                # (Ctrl+Backspace) so one tiny twist kills a whole token.
                print(f"[dial] tiny CCW ({ccw}) -> delete word back (ctrl+backspace, psreadline)")
                keyboard.send("ctrl+backspace")
            elif fam == "bash":
                print(f"[dial] tiny CCW ({ccw}) -> undo (ctrl+shift+-, bash)")
                keyboard.send("ctrl+shift+-")
            else:
                print(f"[dial] tiny CCW ({ccw}) -> undo (ctrl+z)")
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

# ---- foreground-process detection (for terminal-aware undo/redo remap) --
_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.QueryFullProcessImageNameW.argtypes = [
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
]
_kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL

# Extra Win32 signatures for focusing Windows Terminal (COIDEA dial press).
_EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
_user32.EnumWindows.restype = wintypes.BOOL
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.restype = wintypes.BOOL
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.ShowWindow.restype = wintypes.BOOL
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.SetForegroundWindow.restype = wintypes.BOOL
_SW_RESTORE = 9

# Terminals running bash-style readline where Ctrl+Z suspends the process
# instead of undoing. We swap tiny-CCW to the readline undo sequence
# (Ctrl+X, Ctrl+U) and skip tiny-CW since readline has no redo.
# Windows Terminal + PowerShell uses PSReadLine which handles Ctrl+Z/Ctrl+Y
# natively, so those hosts are NOT listed here.
# Windows Terminal / conhost / pwsh / cmd all route keystrokes to PSReadLine
# when it's active. PSReadLine defaults: Ctrl+Z=Undo, Ctrl+Y=Redo, Esc=RevertLine.
PSREADLINE_TERMINALS = {
    "windowsterminal.exe",
    "openconsole.exe",
    "conhost.exe",
    "pwsh.exe",
    "powershell.exe",
    "cmd.exe",
}
# Mintty runs bash/zsh via readline: Ctrl+Z suspends (bad), Ctrl+Y yanks (bad).
# Use Ctrl+_ (Ctrl+Shift+-) for readline undo; clear with Ctrl+A then Ctrl+K.
BASH_TERMINALS = {
    "mintty.exe",            # Git Bash / Cygwin
}


def _foreground_process_name():
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return None
    h = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if not _kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return None
        return os.path.basename(buf.value).lower()
    finally:
        _kernel32.CloseHandle(h)


def _shell_family():
    name = _foreground_process_name()
    if not name:
        return None
    if name in PSREADLINE_TERMINALS:
        return "psreadline"
    if name in BASH_TERMINALS:
        return "bash"
    return None


# ---- COIDEA macropad dial: tab-switch-mode + undo/redo -------------------
def _find_windows_terminal_hwnd():
    result = [0]

    def cb(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return True
        h = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not h:
            return True
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if _kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                if os.path.basename(buf.value).lower() == "windowsterminal.exe":
                    result[0] = hwnd
                    return False  # stop enumeration
        finally:
            _kernel32.CloseHandle(h)
        return True

    _user32.EnumWindows(_EnumWindowsProc(cb), 0)
    return result[0]


def _focus_windows_terminal():
    hwnd = _find_windows_terminal_hwnd()
    if not hwnd:
        print("[coidea] Windows Terminal window not found -- launch it first")
        return False
    _user32.ShowWindow(hwnd, _SW_RESTORE)
    # Alt-tap trick to let SetForegroundWindow escape focus-stealing prevention.
    keyboard.press_and_release("alt")
    _user32.SetForegroundWindow(hwnd)
    return True


def _on_coidea_press(e):
    if state["tab_switch_mode"]:
        state["tab_switch_mode"] = False
        print("[coidea] tab-switch OFF")
        return
    if _foreground_process_name() != "windowsterminal.exe":
        _focus_windows_terminal()
    state["tab_switch_mode"] = True
    print("[coidea] tab-switch ON (rotate to switch tabs, press to confirm)")


def _on_coidea_cw(e):
    if state["tab_switch_mode"]:
        keyboard.send("ctrl+tab")
        return
    fam = _shell_family()
    if fam == "bash":
        print("[coidea] CW -> redo SKIPPED (bash)")
    elif fam == "psreadline":
        print("[coidea] CW -> redo SKIPPED (terminal, avoid beep)")
    else:
        keyboard.send("ctrl+y")


def _on_coidea_ccw(e):
    if state["tab_switch_mode"]:
        keyboard.send("ctrl+shift+tab")
        return
    fam = _shell_family()
    if fam == "psreadline":
        keyboard.send("ctrl+backspace")
    elif fam == "bash":
        keyboard.send("ctrl+shift+-")
    else:
        keyboard.send("ctrl+z")


keyboard.on_press_key(COIDEA_DIAL_CW, _on_coidea_cw, suppress=False)
keyboard.on_press_key(COIDEA_DIAL_CCW, _on_coidea_ccw, suppress=False)
keyboard.on_press_key(COIDEA_DIAL_PRESS, _on_coidea_press, suppress=False)


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


if USE_MEDIA_DIAL:
    _media_hook = MediaKeyHook()
    _media_hook.start()
    atexit.register(_media_hook.stop)
    print("dial:   tinyCW=redo  quickCW=record  tinyCCW=undo  quickCCW=empty")
    print("        CCW->CW=correction  CW->CCW=organize  press=send  (F9/F10 still active)")
    print("volume: media keys suppressed while running; normal behavior restored on exit")
else:
    print("media-key dial: disabled (set USE_MEDIA_DIAL=1 to re-enable)")


try:
    keyboard.wait(QUIT_KEY)
except KeyboardInterrupt:
    pass
print("bye")
