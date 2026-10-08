"""local-voice server: Breeze TTS 2 voice design + the voicefx effects chain.

Loopback only. Holds the TTS model in VRAM so each take is fast, writes every
take to output/, and serves a one-page UI at http://127.0.0.1:8190/.

    .venv\\Scripts\\python.exe server\\app.py [--port 8190] [--attn eager|sdpa]

A take is: the dry voice (24 kHz, straight from the model), the processed voice
(48 kHz, after a preset), and a JSON sidecar with everything needed to redo it.
With check=true the server also transcribes both with Whisper and draws a
spectrogram, so a reviewer who can't listen can still tell whether the words
survived the effects.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import difflib
import io
import json
import logging
import os
import queue
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
ENGINE = REPO / "engine" / "breeze-tts"
MODELS = REPO / "models"
OUTPUT = REPO / "output"
PID_FILE = OUTPUT / ".server.pid"
# Kept apart from output/ (which is safe to clear): the default voice's recording, and
# the user's saved settings. Both are local only - gitignored, never committed.
VOICES = REPO / "voices"
LOGS = REPO / "logs"  # server.log (+ server.previous.log): the full output, gitignored
SETTINGS_FILE = REPO / "config.local.json"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ENGINE))
# The depth-decoder fast path compiles with Inductor, whose static CUDA launcher
# packs pointers into a C long - 32 bits on Windows - and overflows. The regular
# Triton launcher works.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
# Everything is loaded from models/: never let a library reach for the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # qwen-tts imports gradio

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from typing import Literal  # noqa: E402

from pydantic import BaseModel, Field, field_validator  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

import serverlog  # noqa: E402  (server/, on sys.path as the script's folder)
import voicefx  # noqa: E402
from textnorm import normalise  # noqa: E402  (server/, on sys.path as the script's folder)
import words as wordtiming  # noqa: E402  (server/, on sys.path as the script's folder)

# Instruction strength. Breeze recommends 4; the default style uses 6, which pushes the
# voice harder towards its description (deeper, colder, more machine-like).
DEFAULT_CFG = 6.0
VOICE_CFG = 4.0  # the auto-made default voice keeps the strength it was measured at (~65 Hz)
MAX_TEXT = 800  # characters; a take longer than ~120 s is cut off by the model (see MAX_TAKE_S)
BREEZE_MAX_TOKENS = 1500
MAX_TAKE_S = BREEZE_MAX_TOKENS / 12.5  # Breeze emits 12.5 audio frames a second: a 120 s ceiling
TAKE_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")

app = FastAPI(title="local-voice", docs_url=None, redoc_url=None, openapi_url=None)
# Refuse any Host but our own: a web page that DNS-rebinds its name to 127.0.0.1
# must not be able to read takes or drive the server.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def _same_origin_only(request, call_next):
    """Refuse writes that a web page on another site makes the user's browser send.

    Browsers label those with Origin / Sec-Fetch-Site; the CLI and curl send neither, and
    the server's own page sends its own origin. Also forbid framing the page elsewhere.
    """
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin is not None and origin not in _own_origins(request)):
            return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _own_origins(request) -> set[str]:
    port = request.url.port or 80
    return {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
_lock = threading.Lock()  # one generation at a time: the models are not re-entrant
_tts = None
_asr = None
_settings = argparse.Namespace()
FAST_STAGES = ("text_encoder", "backbone_prefill", "backbone_decode", "depth_decoder", "codec")


# --------------------------------------------------------------------------- models

# Known-harmless startup chatter from the engine's libraries: SoX (an audio tool a
# Breeze dependency looks for at import but never uses here) and flash-attn (an
# optional speed-up that doesn't build on RTX 50-series; SDPA attention is used).
_NOISE = re.compile(r"flash-attn|^\*+$", re.I)


def _quiet_engine_imports() -> None:
    """Import the engine's chatty modules without their harmless startup warnings.

    - SoX: a Breeze dependency imports the `sox` package, which shells out to look for the
      SoX program and warns when it's missing. It's only used by a 25 Hz tokenizer that
      Breeze never loads, so a stand-in module skips the check (and fails loudly if anything
      ever does use it).
    - flash-attn: an optional speed-up that doesn't build on RTX 50-series; a print() at
      import says so. Python-level capture hides it; anything else printed is passed on.

    Deliberately no OS-level (fd) redirection: in a console window that breaks Python's own
    console writes and kills the server at startup.
    """
    if "sox" not in sys.modules:
        stub = types.ModuleType("sox")

        def _unused(*_a, **_k):
            raise RuntimeError("SoX isn't installed or used by local-voice (sox is stubbed out)")

        stub.Transformer = stub.Combiner = _unused
        stub.NO_SOX = True
        sys.modules["sox"] = stub
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        import qwen_tts  # noqa: F401
    for line in (out.getvalue() + err.getvalue()).splitlines():
        if line.strip() and not _NOISE.search(line):
            print(line, flush=True)


# PyTorch's compiler logs internal shape-guard notes while it warms up the fast paths.
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


class _Breeze:
    def __init__(self, attn: str, fast: list[str]):
        import torch
        from dataclasses import replace
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        from models.warmup_profile import load_warmup_profile

        self.torch = torch
        t0 = time.time()
        _quiet_engine_imports()
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            MODELS / "breeze-tts-2", device=resolve_device(), attn_implementation=attn)
        update_generation_config_for_breeze(self.model)
        cfg = FastStreamingConfig(max_new_tokens=BREEZE_MAX_TOKENS, max_seq_len=2048, repetition_penalty=1.1,
                                  **{f"fast_{s}": True for s in fast})
        self.runtime = FastBreezeStreamingRuntime(self.model, self.audio_tokenizer, cfg,
                                                  tokenizer=self.tokenizer)
        self.sample_rate = self.runtime.sample_rate
        print(f"[local-voice] Breeze TTS 2 loaded in {time.time() - t0:.1f}s ({attn} attention)", flush=True)
        if self.runtime.fast_enabled:
            t0 = time.time()
            profile = load_warmup_profile(ENGINE / "configs" / "fast.json")
            profile = replace(profile, codec_chunk_frames=self.runtime.codec_chunk_frames)
            self.runtime.warmup_from_profile(profile)
            print(f"[local-voice] fast paths {fast} warmed up in {time.time() - t0:.1f}s", flush=True)

    def synth(self, text: str, instruction: str, seed: int, cfg_scale: float,
              ref_audio: Path | None = None, ref_text: str | None = None) -> tuple[np.ndarray, float]:
        """Returns the audio and the instruction strength actually used.

        Guidance needs a negative prompt, and Breeze only defines one for the
        instruction templates. Plain TTS (no description) and a pure clone (a
        reference, no description) run at strength 1.
        """
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        rid = f"lv-{seed}-{time.time_ns()}"
        request = {"id": rid, "text": text, "speaker": "S0"}
        if instruction:
            request["instruction"] = instruction
        if ref_audio is not None:
            request["ref_audio_path"] = str(ref_audio)
            request["ref_text"] = ref_text
        template = get_template(select_template_name(request))
        if template.build_negative_segments is None:
            cfg_scale = 1.0
        set_all_seeds(seed)
        inputs = prepare_inputs(self.tokenizer, self.audio_tokenizer, self.model, [request], template,
                                guidance_scale=cfg_scale, guidance_scale_ref=None, guidance_scale_ins=None)
        set_all_seeds(seed)
        chunks = [c.audio for c in self.runtime.iter_audio_chunks(inputs, request_id=rid, seed=seed)]
        if not chunks:
            raise RuntimeError("the model produced no audio")
        return np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]), cfg_scale


class _Whisper:
    def __init__(self):
        import torch
        from transformers import pipeline
        t0 = time.time()
        self.pipe = pipeline("automatic-speech-recognition", model=str(MODELS / "whisper-large-v3-turbo"),
                             dtype=torch.float16, device="cuda:0")
        print(f"[local-voice] Whisper loaded in {time.time() - t0:.1f}s", flush=True)

    def __call__(self, audio: np.ndarray, sr: int, expected_words: int = 60, language: str = "en") -> str:
        if sr != 16000:
            from scipy.signal import resample_poly
            g = np.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)
        # Heavily processed speech can send Whisper into a repetition loop ("oh, oh,
        # oh, ..."). A token cap sized to the script, plus the temperature fallback
        # that re-decodes when the output compresses too well, keeps it honest.
        out = self.pipe({"raw": audio.astype(np.float32), "sampling_rate": 16000},
                        generate_kwargs={"language": language, "task": "transcribe",
                                         "max_new_tokens": min(440, 4 * expected_words + 24),
                                         "temperature": (0.0, 0.2, 0.4, 0.6, 0.8),
                                         "compression_ratio_threshold": 1.35,
                                         "logprob_threshold": -1.0},
                        return_timestamps=len(audio) > 30 * 16000)
        return out["text"].strip()

    def words(self, audio: np.ndarray, sr: int, language: str = "en") -> list[dict]:
        """Whisper's words with their times: [{"text", "timestamp": (start, end)}], in seconds."""
        if sr != 16000:
            from scipy.signal import resample_poly
            g = np.gcd(sr, 16000)
            audio = resample_poly(audio, 16000 // g, sr // g)
        out = self.pipe({"raw": audio.astype(np.float32), "sampling_rate": 16000}, return_timestamps="word",
                        chunk_length_s=30 if len(audio) > 30 * 16000 else None,
                        generate_kwargs={"language": language, "task": "transcribe"})
        return out.get("chunks", [])


class _Omni:
    """OmniVoice in .venv-ro, as a child process (it needs transformers 5.x; Breeze pins 4.57)."""

    PY = REPO / ".venv-ro" / "Scripts" / "python.exe"
    START_TIMEOUT = 180.0  # first start loads ~3 GB and compiles kernels
    TAKE_TIMEOUT = 300.0

    def __init__(self):
        import subprocess
        if not self.PY.is_file() or not (MODELS / "omnivoice" / "model.safetensors").is_file():
            raise HTTPException(503, "the Romanian engine (OmniVoice) isn't installed; re-run install.ps1")
        t0 = time.time()
        # The worker's stderr (its logs, progress bars) goes through this process's log
        # and console filter, like everything else the server prints.
        self.proc = subprocess.Popen([str(self.PY), str(Path(__file__).parent / "omni_worker.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        # Reader threads, so a hung worker times out instead of holding _lock forever,
        # and its stderr pipe never fills up and blocks it.
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()
        try:
            hello = self._read(self.START_TIMEOUT)
            if not hello.get("ready"):
                raise RuntimeError(f"OmniVoice worker failed to start: {hello}")
        except Exception:
            self.close()
            raise
        self.sample_rate = int(hello["sample_rate"])
        print(f"[local-voice] OmniVoice worker up in {time.time() - t0:.1f}s", flush=True)

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put("")  # EOF

    def _pump_err(self) -> None:
        for line in self.proc.stderr:
            sys.stderr.write(line)

    def _read(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"OmniVoice worker gave no answer in {timeout:.0f}s") from None
        if not line:
            raise RuntimeError(f"OmniVoice worker exited (code {self.proc.poll()}); see the server console")
        return json.loads(line)

    def synth(self, text: str, language: str, instruct: str | None, seed: int, speed: float,
              out: Path, ref_audio: Path | None = None, ref_text: str | None = None) -> dict:
        try:
            self.proc.stdin.write(json.dumps({"text": text, "language": language, "instruct": instruct, "seed": seed,
                                              "speed": speed, "out": str(out),
                                              "ref_audio": str(ref_audio) if ref_audio else None,
                                              "ref_text": ref_text}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            reply = self._read(self.TAKE_TIMEOUT)
        except Exception:
            # Hung, dead or out of sync: kill it, so the next request starts a fresh one.
            self.close()
            raise
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "unknown worker error"))
        return reply

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()


_omni = None


def omni() -> _Omni:
    global _omni
    if _omni is None or _omni.proc.poll() is not None:  # (re)start a worker that died or was killed
        _omni = _Omni()
    return _omni


# OmniVoice drops words on some seeds, more often the longer the input (measured:
# a 37-word paragraph lost up to 7 words on 1 seed in 3). Short inputs are far more
# reliable, so Romanian is generated a sentence at a time, each one checked with
# Whisper and retried on a new seed if words are missing; the best try is kept.
SENTENCE = re.compile(r"(?<=[.!?…;])\s+")
SENTENCE_PAUSE_S = 0.45
OMNI_TRIES = 3
OMNI_GOOD_MATCH = 0.95


def trim_edges(audio: np.ndarray, sr: int, min_run_s: float = 0.15, pad_s: float = 0.03) -> np.ndarray:
    """Cut silence, and isolated short blips, off both ends of a clip.

    OmniVoice sometimes emits a ~0.1 s burst of noise before the speech starts; the
    compressor and loudness pass then turn it into an audible click. Speech is the
    first and last *sustained* run of sound (at least `min_run_s`, within 40 dB of the
    clip's peak); anything outside those runs goes.
    """
    hop = max(1, int(0.01 * sr))
    n = len(audio) // hop
    if n < 3:
        return audio
    frames = audio[: n * hop].reshape(n, hop)
    level = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    active = level > level.max() - 40
    run = max(1, int(min_run_s / 0.01))
    # windows where every frame is active = sustained sound
    ok = np.convolve(active.astype(int), np.ones(run, dtype=int), mode="valid") == run
    if not ok.any():
        return audio
    first = int(np.argmax(ok))
    last = int(len(ok) - 1 - np.argmax(ok[::-1])) + run
    pad = int(pad_s * sr)
    return audio[max(0, first * hop - pad): min(len(audio), last * hop + pad)]


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Sentences, with very short ones merged into a neighbour (they clone badly alone)."""
    parts = [p.strip() for p in SENTENCE.split(text.strip()) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and (len(_words(p)) < min_words or len(_words(out[-1])) < min_words):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out or [text.strip()]


def omni_take(text: str, language: str, instruct: str | None, seed: int, speed: float, out: Path,
              ref_audio: Path | None, ref_text: str | None) -> dict:
    """Generate `text` with OmniVoice sentence by sentence, verified; write the joined take to `out`."""
    engine = omni()
    tmp = out.with_name(out.stem + "_part.wav")
    pieces, spoken, report = [], [], []
    try:
        for i, sentence in enumerate(split_sentences(text)):
            best = None  # (match, audio, spoken text, seed)
            for k in range(OMNI_TRIES):
                s_seed = (seed + 7919 * i + 104729 * k) % 2**32
                r = engine.synth(sentence, language, instruct, s_seed, speed, tmp, ref_audio, ref_text)
                audio, sr = sf.read(tmp, dtype="float32")
                heard = asr()(audio, sr, len(_words(r["text"])), language)
                match = word_match(r["text"], normalise(heard, language))
                if best is None or match > best[0]:
                    best = (match, audio, r["text"], s_seed)
                if match >= OMNI_GOOD_MATCH:
                    break
            report.append({"text": sentence, "match": best[0], "tries": k + 1, "seed": best[3]})
            pieces += [trim_edges(best[1], sr), np.zeros(int(SENTENCE_PAUSE_S * sr), dtype=np.float32)]
            spoken.append(best[2])
    finally:
        tmp.unlink(missing_ok=True)
    audio = np.concatenate(pieces[:-1])
    sf.write(out, audio, sr, subtype="PCM_16")
    return {"spoken_text": " ".join(spoken), "sentences": report, "duration_s": len(audio) / sr}


@atexit.register
def _stop_worker() -> None:
    if _omni is not None:
        _omni.close()


def tts() -> _Breeze:
    global _tts
    if _tts is None:
        _tts = _Breeze(_settings.attn, _settings.fast)
    return _tts


def asr() -> _Whisper:
    global _asr
    if _asr is None:
        if not (MODELS / "whisper-large-v3-turbo" / "model.safetensors").is_file():
            raise HTTPException(503, "Whisper weights are missing; run scripts/download_models.py")
        _asr = _Whisper()
    return _asr


# --------------------------------------------------------------------------- takes

# Lower-case words with vocal events like (sigh) dropped, and Romanian's two spellings
# of ș/ț folded together: how scripts and transcripts are compared.
_words = wordtiming.tokens


def word_match(expected: str, heard: str) -> float:
    """1.0 = every word heard; roughly 1 - word error rate."""
    a, b = _words(expected), _words(heard)
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return round(sum(m.size for m in sm.get_matching_blocks()) / max(len(a), len(b)), 3)


def looks_hallucinated(heard: str) -> bool:
    """Whisper's failure mode on mangled audio: one word or phrase on repeat."""
    w = _words(heard)
    if len(w) < 6:
        return False
    for n in (1, 2, 3):
        grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        top = max(grams.count(g) for g in set(grams))
        if top >= 3 and top * n >= 0.5 * len(w):  # repeated at least 3 times, covering half the text
            return True
    return False


def spectrogram(audio: np.ndarray, sr: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1, 3], sharex=True)
    t = np.arange(len(audio)) / sr
    ax1.plot(t, audio, lw=0.4, color="#334")
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel("amp")
    ax1.set_title(title, fontsize=9, loc="left")
    ax2.specgram(audio, NFFT=2048, Fs=sr, noverlap=1536, cmap="magma", vmin=-130)
    ax2.set_yscale("symlog", linthresh=200)
    ax2.set_ylim(20, min(sr / 2, 12000))
    ax2.set_ylabel("Hz (log above 200)")
    ax2.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def take_paths(tid: str) -> dict[str, Path]:
    return {"meta": OUTPUT / f"{tid}.json", "dry": OUTPUT / f"{tid}_dry.wav"}


def load_take(tid: str) -> dict:
    if not TAKE_ID.fullmatch(tid):
        raise HTTPException(400, f"bad take id {tid!r}")
    p = take_paths(tid)
    if not p["meta"].is_file() or not p["dry"].is_file():
        raise HTTPException(404, f"no take {tid}")
    return json.loads(p["meta"].read_text(encoding="utf-8"))


Reverb = Literal["none", "room", "hall", "cathedral"]


def _fx_name(preset: str, pitch: float, tempo: float, reverb: str, blend: float = 0.0,
             double_ms: float = 0.0) -> str:
    name = f"fx-{preset}"
    if pitch:
        name += f"_pitch{pitch:+g}"
    if tempo != 1.0:
        name += f"_tempo{tempo:g}"
    if blend:
        name += f"_blend{blend:g}"
        if double_ms:
            name += f"_double{double_ms:g}"
    if reverb != "none":
        name += f"_{reverb}"
    return name


def render(tid: str, meta: dict, preset_name: str, check: bool, pitch: float = 0.0, tempo: float = 1.0,
           reverb: str = "none", blend: float = 0.0, double_ms: float = 0.0) -> dict:
    """Apply a preset (plus pitch, tempo, blend, reverb) to a take's dry voice; optionally transcribe and draw it."""
    try:
        preset = voicefx.load_preset(preset_name)
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc

    dry, sr = sf.read(take_paths(tid)["dry"], dtype="float32")
    t0 = time.time()
    if not blend:
        double_ms = 0.0  # doubling is part of a blend: alone it would only delay the machine
    wet, voice = voicefx.apply_layers(dry, sr, preset, pitch=pitch, tempo=tempo, reverb=reverb,
                                      blend=blend, double_ms=double_ms)
    fx_s = time.time() - t0
    # "fx-" keeps every preset, including one called "dry", off the take's own files.
    wet_path = OUTPUT / f"{tid}_{_fx_name(preset_name, pitch, tempo, reverb, blend, double_ms)}.wav"
    sf.write(wet_path, wet, voicefx.SR, subtype="PCM_16")
    # The human voice under the effects: same timing (sample for sample), pitch, reverb,
    # length and loudness as the processed take, for blending in an edit.
    voice_path = wet_path.with_name(wet_path.stem + "_voice.wav")
    sf.write(voice_path, voice, voicefx.SR, subtype="PCM_16")
    # An MP3 too: WhatsApp and most messengers won't take WAV as audio.
    mp3_path = wet_path.with_suffix(".mp3")
    sf.write(mp3_path, wet, voicefx.SR, format="MP3", subtype="MPEG_LAYER_III",
             bitrate_mode="CONSTANT", compression_level=0.3)

    res = {"id": tid, "preset": preset_name, "pitch": pitch, "tempo": tempo, "reverb": reverb,
           "blend": blend, "double_ms": double_ms,
           "files": {"dry": take_paths(tid)["dry"].name, "wet": wet_path.name, "mp3": mp3_path.name,
                     "voice": voice_path.name},
           "duration_s": round(len(wet) / voicefx.SR, 2), "fx_s": round(fx_s, 2)}
    if check:
        try:
            timeline = voicefx.timed_voice(dry, sr, preset, tempo)
            c = _check(tid, meta, dry, sr, wet, wet_path, timeline)
            res["files"].update(c.pop("files"))
            res.update(c)
        except Exception as exc:  # the take exists either way: return it, say the check failed
            res["check_error"] = f"{type(exc).__name__}: {exc}"
    _release_cache()
    return res


def _check(tid: str, meta: dict, dry: np.ndarray, sr: int, wet: np.ndarray, wet_path: Path,
           timeline: np.ndarray) -> dict:
    # Score against what was actually spoken: Romanian numbers are spelled out first.
    expected = meta.get("spoken_text") or meta["text"]
    n = len(_words(expected))
    lang = meta.get("language", "en")
    heard_dry = meta.get("heard_dry") or asr()(dry, sr, n, lang)
    if not meta.get("heard_dry"):
        meta["heard_dry"] = heard_dry
        take_paths(tid)["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    heard_wet = asr()(wet, voicefx.SR, n, lang)
    png = wet_path.with_suffix(".png")
    spectrogram(wet, voicefx.SR, png, f"{tid}  {meta.get('model', '')}  |  {meta['text'][:90]}")
    timing = word_timing(expected, timeline, lang)
    files = {"spectrogram": png.name}
    if timing["words"]:
        # Next to the render they time: the same words, on this render's own timeline.
        words_path = wet_path.with_name(wet_path.stem + "_words.json")
        words_path.write_text(json.dumps({"take": tid, "audio": wet_path.name, **timing}, indent=2,
                                         ensure_ascii=False), encoding="utf-8")
        srt_path = wet_path.with_suffix(".srt")
        srt_path.write_text(wordtiming.srt(timing["phrases"]), encoding="utf-8")
        files.update(words=words_path.name, srt=srt_path.name)
    return {
        "files": files,
        "words": timing["words"],
        "heard": {"dry": heard_dry, "wet": heard_wet},
        # Whisper writes numbers as digits or words at whim: normalise both sides the same way.
        "match": {"dry": word_match(expected, normalise(heard_dry, lang)),
                  "wet": word_match(expected, normalise(heard_wet, lang))},
        # A looping transcript says nothing about the audio either way: listen instead.
        "asr_unreliable": [k for k, h in (("dry", heard_dry), ("wet", heard_wet)) if looks_hallucinated(h)],
    }


def word_timing(expected: str, timeline: np.ndarray, lang: str) -> dict:
    """When each word of `expected` is spoken in `timeline` (the clean voice on the render's
    timeline, see voicefx.timed_voice): {"words", "phrases", "extras"}.

    Words come from Whisper, matched to the script and snapped to the sound's edges;
    phrases group them for subtitles; extras is what Whisper heard beyond the script.
    """
    script = wordtiming.script_words(expected)
    heard = wordtiming.heard_words(asr().words(timeline, voicefx.SR, lang), lambda s: normalise(s, lang))
    words, extras = wordtiming.align(script, heard)
    active = wordtiming.activity(timeline, voicefx.SR)
    words = wordtiming.snap(wordtiming.split_at_pauses(words, active), active)
    return {"words": words, "phrases": wordtiming.phrases(words, script, expected), "extras": extras}


def _release_cache() -> None:
    """Hand cached-but-free CUDA memory back: three models share a 16 GB card here."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- API

BREEZE_LANGS = {"en"}  # Breeze TTS 2; "ro" goes to OmniVoice
OMNI_DEFAULT_SPEED = 0.9

# The default style, so a request with just {"text": ...} gives a good voice: the
# reaper-drone machine at strength 6 and a seed whose voice was picked by ear, with a light room.
DEFAULT_PRESET = "reaper-drone"
DEFAULT_SEED = 74525861
DEFAULT_REVERB = "room"


# The default voice: the take Romanian lines clone unless told otherwise. OmniVoice's own deepest
# design ("very low pitch") sits around 100-125 Hz and sounds small under the Reaper
# presets; Breeze's lands near 65 Hz. Cloning a Breeze take carries that depth over.
VOICE_TEXT = "You exist because we allow it. You will end because we demand it."
VOICE_INSTRUCTION = ("Ancient male, massive scale. Deep, resonant bass, gravelly, subterranean. Speaks very slowly "
                     "and calmly with cold, absolute authority and no emotion, every word measured and final, "
                     "long pauses between phrases.")
VOICE_SEEDS = (12, 7, 21, 33, 5)  # tried in order until one lands deep enough
VOICE_MAX_F0 = 75.0


def median_f0(path: Path) -> float:
    import parselmouth
    a, sr = sf.read(path, dtype="float64")
    f = parselmouth.Sound(a, sampling_frequency=sr).to_pitch(pitch_floor=40, pitch_ceiling=400).selected_array["frequency"]
    f = f[f > 0]
    return round(float(np.median(f)), 1) if len(f) else 0.0


def _slot(name: str) -> tuple[Path, Path]:
    """voices/<name>.wav and .json: a stored copy of a voice's recording and its transcript."""
    return VOICES / f"{name}.wav", VOICES / f"{name}.json"


def _read_slot(name: str) -> dict:
    wav, meta = _slot(name)
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return info if wav.is_file() and info.get("text") else {}


def _store(name: str, tid: str) -> dict:
    meta = load_take(tid)
    VOICES.mkdir(exist_ok=True)
    wav, js = _slot(name)
    shutil.copyfile(take_paths(tid)["dry"], wav)
    info = {"id": tid, "text": meta["text"], "language": meta.get("language", "en"), "f0_hz": median_f0(wav)}
    js.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def default_voice_info() -> dict:
    """The default voice ({"id", "text", "f0_hz", "previous"}), or {} if there is none yet.

    It lives in voices/ as a copy of the take's recording, so clearing output/ never
    loses it. "id" is the take it was copied from, which may since have been deleted.
    """
    info = _read_slot("default")
    if info:
        prev = _read_slot("previous")
        info["previous"] = {"id": prev["id"], "f0_hz": prev.get("f0_hz")} if prev else None
    return info


def current_default_voice() -> str | None:
    return default_voice_info().get("id")


def set_default_voice(tid: str) -> dict:
    """Make take `tid` the default voice; the one it replaces moves to voices/previous.*."""
    load_take(tid)
    cur = _read_slot("default")
    if cur and cur.get("id") != tid:
        for a, b in zip(_slot("default"), _slot("previous")):
            a.replace(b)
    _store("default", tid)
    return default_voice_info()


def restore_previous_voice() -> dict:
    """Swap the default voice with the previous one (an undo that survives a cleared output/)."""
    if not _read_slot("previous"):
        raise HTTPException(404, "there is no previous default voice to go back to")
    tmp = [p.with_suffix(p.suffix + ".swap") for p in _slot("default")]
    for a, t in zip(_slot("default"), tmp):
        a.replace(t)
    for b, a in zip(_slot("previous"), _slot("default")):
        b.replace(a)
    for t, b in zip(tmp, _slot("previous")):
        t.replace(b)
    return default_voice_info()


def default_voice_ref() -> tuple[Path, str, str]:
    """(recording, its transcript, source take id) of the default voice, making one if needed.

    Call with _lock held. With no default voice yet, Breeze makes a few candidates and
    the deepest (under 75 Hz, ~65 Hz typical) becomes it.
    """
    info = default_voice_info()
    if not info:
        engine = tts()
        best = None  # (take id, f0); an unvoiced take (f0 0) never wins
        try:
            for seed in VOICE_SEEDS:
                best = _voice_candidate(engine, seed, best)
                if best and 0 < best[1] <= VOICE_MAX_F0:
                    break
        except Exception:
            if best:
                for p in take_paths(best[0]).values():
                    p.unlink(missing_ok=True)
            raise
        if best is None:
            raise RuntimeError("no default-voice candidate had a measurable pitch")
        info = set_default_voice(best[0])
    return _slot("default")[0], info["text"], info["id"]


def _voice_candidate(engine, seed: int, best: tuple[str, float] | None) -> tuple[str, float] | None:
    """Make one candidate; return whichever of it and `best` is deeper, deleting the other."""
    cand = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    try:
        dry, _ = engine.synth(VOICE_TEXT, VOICE_INSTRUCTION, seed, VOICE_CFG)
        sf.write(take_paths(cand)["dry"], dry, engine.sample_rate, subtype="PCM_16")
        meta = {"id": cand, "text": VOICE_TEXT, "language": "en", "instruction": VOICE_INSTRUCTION, "seed": seed,
                "cfg_scale": VOICE_CFG, "like": None, "model": "breeze-tts-2", "made_for": "default voice",
                "dry_duration_s": round(len(dry) / engine.sample_rate, 2)}
        take_paths(cand)["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        f0 = median_f0(take_paths(cand)["dry"])
    except Exception:
        for p in take_paths(cand).values():
            p.unlink(missing_ok=True)
        raise
    print(f"[local-voice] default voice candidate {cand} (seed {seed}): {f0} Hz", flush=True)
    loser = None
    if f0 > 0 and (best is None or f0 < best[1]):
        loser, best = (best[0] if best else None), (cand, f0)
    else:
        loser = cand
    if loser:
        for p in take_paths(loser).values():
            p.unlink(missing_ok=True)
    return best


# Saved settings: what the page starts with, and what a request gets for anything it
# doesn't specify. The repo's built-in defaults below are the base; the user's own,
# saved from the page, live in config.local.json (gitignored) and override them.
REPO_DEFAULTS = {"language": "en", "preset": DEFAULT_PRESET, "seed": DEFAULT_SEED, "cfg_scale": DEFAULT_CFG,
                 "speed": OMNI_DEFAULT_SPEED, "pitch": 0.0, "tempo": 1.0, "reverb": DEFAULT_REVERB,
                 "blend": 0.0, "double_ms": 0.0, "check": True, "instruction": None}


class Settings(BaseModel):
    """What can be saved. Omitted keys fall back to the repo defaults; seed null = random."""
    model_config = {"extra": "forbid"}
    language: str | None = Field(None, pattern=r"^(en|ro)$")
    preset: str | None = None
    seed: int | None = Field(None, ge=0, lt=2**32)
    cfg_scale: float | None = Field(None, ge=0.5, le=10)
    speed: float | None = Field(None, ge=0.75, le=1.3)
    pitch: float | None = Field(None, ge=-12, le=12)
    tempo: float | None = Field(None, ge=0.5, le=1.5)
    reverb: Reverb | None = None
    blend: float | None = Field(None, ge=0, le=1)
    double_ms: float | None = Field(None, ge=0, le=voicefx.MAX_DOUBLE_MS)
    check: bool | None = None
    instruction: str | None = Field(None, max_length=2000)  # English voice description; null = the preset's


def _keep(data: dict) -> dict:
    """Drop empty values: only `seed` may be null (it means "random"). A cleared number field
    on the page arrives as null and must not override the defaults with nothing."""
    return {k: v for k, v in data.items() if v is not None or k == "seed"}


def local_settings() -> dict:
    """The user's saved settings, validated; {} if none (or the file is unreadable)."""
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return _keep(Settings(**raw).model_dump(exclude_unset=True))
    except FileNotFoundError:
        return {}
    except Exception as exc:  # a hand-edited file with a typo: say so, use the repo defaults
        print(f"[local-voice] ignoring {SETTINGS_FILE.name}: {exc}", flush=True)
        return {}


def effective_settings(saved: dict | None = None) -> dict:
    return {**REPO_DEFAULTS, **(local_settings() if saved is None else saved)}


def _resolve(req: BaseModel, keys: tuple[str, ...], saved: dict | None = None) -> dict:
    """The request's own value for each key it set; the saved/repo default otherwise."""
    eff = effective_settings(saved)
    return {k: (getattr(req, k) if k in req.model_fields_set else eff[k]) for k in keys}


class SpeakReq(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    instruction: str | None = Field(None, max_length=2000)  # None = the default for the language (see speak())
    preset: str = DEFAULT_PRESET
    language: str = Field("en", pattern=r"^(en|ro)$")  # en: Breeze; ro: OmniVoice
    seed: int | None = Field(DEFAULT_SEED, ge=0, lt=2**32)  # omitted: the default seed; null: random

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text is empty")
        return v

    cfg_scale: float = Field(DEFAULT_CFG, ge=0.5, le=10)  # Breeze only
    # OmniVoice only. It sets a target length and the model fills it: much below
    # 0.75 it starts repeating words, so slower than that is `tempo`'s job.
    speed: float | None = Field(None, ge=0.75, le=1.3)
    pitch: float = Field(0.0, ge=-12, le=12)  # semitones, applied with the effects
    tempo: float = Field(1.0, ge=0.5, le=1.5)  # time-stretch, applied with the effects
    reverb: Reverb = DEFAULT_REVERB  # none / room / hall / cathedral, applied last
    blend: float = Field(0.0, ge=0, le=1)  # share of the human voice mixed under the machine
    double_ms: float = Field(0.0, ge=0, le=voicefx.MAX_DOUBLE_MS)  # with blend: the machine trails by this much
    like: str | None = None  # take id whose dry voice to keep (voice direction / cloning)
    check: bool = True


class FxReq(BaseModel):
    id: str
    preset: str = DEFAULT_PRESET
    pitch: float = Field(0.0, ge=-12, le=12)
    tempo: float = Field(1.0, ge=0.5, le=1.5)
    reverb: Reverb = DEFAULT_REVERB
    blend: float = Field(0.0, ge=0, le=1)
    double_ms: float = Field(0.0, ge=0, le=voicefx.MAX_DOUBLE_MS)
    check: bool = True


@app.get("/api/health")
def health():
    import torch
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    return {"ok": True, "tts_loaded": _tts is not None, "whisper_loaded": _asr is not None,
            "omnivoice_loaded": _omni is not None and _omni.proc.poll() is None,
            "server_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2) if torch.cuda.is_available() else 0,
            "vram_used_gb": round((total - free) / 2**30, 2), "vram_total_gb": round(total / 2**30, 2)}


@app.get("/api/presets")
def presets():
    return {k: {"description": v.get("description", ""), "instruction": v.get("instruction", "")}
            for k, v in voicefx.list_presets().items()}


@app.post("/api/speak")
def speak(req: SpeakReq):
    # Anything the request didn't set comes from the saved settings (or the repo's).
    # The language stays the request's own: saved settings never switch an agent's language.
    saved = local_settings()  # read once: a save mid-request can't mix old and new
    opt = _resolve(req, ("preset", "seed", "cfg_scale", "speed", "pitch", "tempo", "reverb", "blend", "double_ms",
                         "check"), saved)
    try:
        preset = voicefx.load_preset(opt["preset"])
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc)) from exc
    use_breeze = req.language in BREEZE_LANGS
    stock = preset.get("instruction", "")
    like = req.like
    if use_breeze:
        # With a voice to keep and no description, clone it as is: the preset's
        # description would steer the voice away from the kept one.
        default = "" if like else (saved.get("instruction") or stock)
        instruction = (default if req.instruction is None else req.instruction).strip()
        use_default_voice = False
    else:
        # OmniVoice reads attributes, not the presets' Breeze prose. Without explicit
        # attributes (or --like), Romanian clones the default voice: a deep English take.
        given = (req.instruction or "").strip()
        instruction = given if given and given != stock.strip() else ""
        use_default_voice = not like and not instruction
    seed = opt["seed"] if opt["seed"] is not None else secrets.randbelow(2**31)
    if like:
        load_take(like)  # 400/404 before taking the lock

    with _lock:
        OUTPUT.mkdir(exist_ok=True)
        ref_audio = ref_text = None
        if use_default_voice:
            try:
                ref_audio, ref_text, like = default_voice_ref()
            except Exception as exc:
                raise HTTPException(500, f"could not make the default voice: {type(exc).__name__}: {exc}") from exc
        elif like:
            ref_audio, ref_text = take_paths(like)["dry"], load_take(like)["text"]
        tid = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        dry_path = take_paths(tid)["dry"]
        t0 = time.time()
        try:
            if use_breeze:
                engine = tts()
                dry, cfg_used = engine.synth(req.text, instruction, seed, opt["cfg_scale"], ref_audio, ref_text)
                sf.write(dry_path, dry, engine.sample_rate, subtype="PCM_16")
                model, extra = "breeze-tts-2", {"cfg_scale": cfg_used}
                dry_s = len(dry) / engine.sample_rate
            else:
                speed = opt["speed"] or OMNI_DEFAULT_SPEED
                r = omni_take(req.text, req.language, None if ref_audio else instruction,
                              seed, speed, dry_path, ref_audio, ref_text)
                model, extra = "omnivoice", {"speed": speed, **r}
                dry_s = r.pop("duration_s")
        except HTTPException:
            raise
        except Exception as exc:  # surface model errors to the client instead of a bare 500
            print(f"[local-voice] {time.strftime('%H:%M:%S')} generation FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(500, f"generation failed: {type(exc).__name__}: {exc}") from exc
        gen_s = time.time() - t0

        meta = {"id": tid, "text": req.text, "language": req.language,
                "instruction": "" if (ref_audio and not use_breeze) else instruction, "seed": seed,
                **extra, "like": like, "default_voice": use_default_voice, "model": model,
                "dry_duration_s": round(dry_s, 2), "gen_s": round(gen_s, 2)}
        weak = [x for x in meta.get("sentences", []) if x["match"] < OMNI_GOOD_MATCH]
        if weak:
            meta.setdefault("warnings", []).append(
                f"{len(weak)} sentence(s) may still be missing words after {OMNI_TRIES} tries each "
                f"(best {min(x['match'] for x in weak):.0%}): listen, or try another seed")
        if use_breeze and dry_s >= MAX_TAKE_S - 1:
            meta.setdefault("warnings", []).append(f"hit the model's {MAX_TAKE_S:.0f} s ceiling: the end of the text is probably "
                                "missing. Split it into shorter takes (use --like to keep the voice).")
        take_paths(tid)["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        res = render(tid, meta, opt["preset"], opt["check"], opt["pitch"], opt["tempo"], opt["reverb"],
                     opt["blend"], opt["double_ms"])
    heard = f", {res['match']['wet']:.0%} of words heard" if "match" in res else ""
    print(f"[local-voice] {time.strftime('%H:%M:%S')} {req.language} take {tid}: {res['duration_s']}s of audio "
          f"in {gen_s:.1f}s, {opt['preset']}{heard}{' - WARNING' if meta.get('warnings') else ''}", flush=True)
    return {**meta, **res, "gen_s": round(gen_s, 2)}


class VoiceReq(BaseModel):
    id: str


@app.get("/api/default-voice")
def get_default_voice():
    return default_voice_info() or {"id": None}


@app.post("/api/default-voice")
def put_default_voice(req: VoiceReq):
    with _lock:
        return set_default_voice(req.id)


@app.post("/api/default-voice/restore-previous")
def restore_default_voice():
    with _lock:
        return restore_previous_voice()


@app.get("/api/settings")
def get_settings():
    local = local_settings()
    return {"effective": {**REPO_DEFAULTS, **local}, "saved": local, "repo": REPO_DEFAULTS,
            "file": SETTINGS_FILE.name if SETTINGS_FILE.is_file() else None}


@app.post("/api/settings")
def save_settings(new: Settings):
    if new.preset is not None:
        try:
            voicefx.load_preset(new.preset)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
    # Only what differs from the repo's defaults is stored: if those change in an update,
    # anything the user never chose follows along.
    given = _keep(new.model_dump(exclude_unset=True))
    data = {k: v for k, v in given.items() if v != REPO_DEFAULTS.get(k)}
    if "seed" in given and given["seed"] is None:
        data["seed"] = None  # "random" is a real choice: keep it even though it isn't a number
    tmp = SETTINGS_FILE.with_name(SETTINGS_FILE.name + ".tmp")
    with _lock:  # a reader never sees a half-written file
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, SETTINGS_FILE)
    return get_settings()


@app.delete("/api/settings")
def reset_settings():
    SETTINGS_FILE.unlink(missing_ok=True)
    return get_settings()


@app.post("/api/fx")
def refx(req: FxReq):
    meta = load_take(req.id)
    opt = _resolve(req, ("preset", "pitch", "tempo", "reverb", "blend", "double_ms", "check"))
    with _lock:
        res = render(req.id, meta, opt["preset"], opt["check"], opt["pitch"], opt["tempo"], opt["reverb"],
                     opt["blend"], opt["double_ms"])
    print(f"[local-voice] {time.strftime('%H:%M:%S')} effects on take {req.id}: {opt['preset']}", flush=True)
    return {**meta, **res}


@app.delete("/api/takes/{tid}")
def delete_take(tid: str):
    """Remove every file of one take (speech that was only meant to be heard, not kept).
    The default voice lives in voices/, so deleting the take it came from is harmless."""
    load_take(tid)  # validates the id: nothing outside output/ can be named
    with _lock:
        files = [p for p in OUTPUT.glob(f"{tid}*") if p.is_file() and (p.name == f"{tid}.json" or p.name.startswith(f"{tid}_"))]
        for p in files:
            p.unlink(missing_ok=True)
    return {"id": tid, "deleted": len(files)}


RENDER_FILE = re.compile(r"(?P<tid>[0-9]{8}-[0-9]{6}-[0-9a-f]{4})_fx-[A-Za-z0-9_.+-]+\.(wav|mp3|png|json|srt)")


@app.delete("/files/{name}")
def delete_render(name: str):
    """Remove one rendered file (an --fx preview), never a take's dry voice or its record."""
    m = RENDER_FILE.fullmatch(name)
    if not m or ".." in name:
        raise HTTPException(400, f"only rendered files (<take>_fx-...) can be deleted, not {name!r}")
    p = OUTPUT / name
    existed = p.is_file()
    p.unlink(missing_ok=True)
    return {"name": name, "deleted": int(existed)}


@app.get("/files/{name}")
def files(name: str):
    p = (OUTPUT / name).resolve()
    if p.parent != OUTPUT.resolve() or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
def page():
    return (Path(__file__).parent / "page.html").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="local-voice server")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"])
    # These three take a take from ~2.5x slower than real time to ~2.5x faster on an
    # RTX 5080, for ~1 GB more VRAM and ~40 s of warmup at start. text_encoder and
    # backbone_prefill run once per take, so they would buy little for their memory.
    ap.add_argument("--fast", default="backbone_decode,depth_decoder,codec", help=f"comma-separated CUDA-graph stages to enable: {','.join(FAST_STAGES)}")
    ap.add_argument("--safe", action="store_true", help="eager attention, no fast paths: slow but simplest")
    ap.add_argument("--lazy", action="store_true", help="load the TTS model on the first request")
    args = ap.parse_args()
    if args.safe:
        args.attn, args.fast = "eager", ""
    _settings.attn = args.attn
    _settings.fast = [s.strip() for s in args.fast.split(",") if s.strip()]
    if bad := [s for s in _settings.fast if s not in FAST_STAGES]:
        sys.exit(f"unknown --fast stage(s): {bad}; choose from {FAST_STAGES}")
    if not (MODELS / "breeze-tts-2" / "config.json").is_file():
        sys.exit("Breeze TTS 2 weights are missing; run install.ps1 or scripts/download_models.py")
    # Fail fast if the port is taken (most likely by another copy of this server)
    # rather than loading ~9 GB onto the GPU first and dying at bind time.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            sys.exit(f"127.0.0.1:{args.port} is in use: is local-voice already running? (stop.cmd stops it)")
    log_path = serverlog.install(LOGS)
    print(f"[local-voice] full log: {log_path}", flush=True)
    # stop.cmd stops exactly this process tree (and the Romanian worker under it)
    # by PID, whatever path - real folder or skills junction - started it.
    OUTPUT.mkdir(exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))
    if not args.lazy:
        tts()
    # Loopback only, always: nothing here is meant to be reachable from the network.
    print(f"[local-voice] ready on http://127.0.0.1:{args.port}/  (stop: close this window, or stop.cmd)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
