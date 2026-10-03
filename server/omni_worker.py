"""OmniVoice worker: the Romanian TTS engine, in its own venv.

OmniVoice needs transformers 5.x and Breeze TTS 2 pins 4.57, so they can't share
a Python environment. The main server starts this script with .venv-ro's Python
and talks to it over stdin/stdout, one JSON object per line:

    -> {"text": ..., "language": "ro", "out": "C:/.../x_dry.wav",
        "instruct": "male, elderly, very low pitch" | null,
        "ref_audio": path | null, "ref_text": str | null, "speed": 0.85, "seed": 123}
    <- {"ok": true, "sample_rate": 24000, "duration_s": 5.2, "text": <normalised>}
    <- {"ok": false, "error": "..."}

It prints {"ready": true} once the model is loaded. Logs go to stderr.
"""

from __future__ import annotations

import os
import sys

# The protocol owns the real stdout. Before anything else can print, keep a private
# handle on it and point file descriptor 1 at stderr, so stray output from Python
# *or* native code (CUDA, Triton, progress bars) lands in the log, not the stream.
_proto = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)
sys.stdout = sys.stderr

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

from textnorm import normalise  # noqa: E402  (same folder)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

MODEL = Path(__file__).resolve().parent.parent / "models" / "omnivoice"

def log(*a) -> None:
    print("[omnivoice]", *a, file=sys.stderr, flush=True)


def send(obj: dict) -> None:
    _proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _proto.flush()


def main() -> None:
    import soundfile as sf
    import torch
    from omnivoice import OmniVoice

    sys.stdin.reconfigure(encoding="utf-8")
    t0 = time.time()
    model = OmniVoice.from_pretrained(str(MODEL), device_map="cuda:0", dtype=torch.float16)
    sr = int(getattr(model, "sampling_rate", 24000))
    log(f"loaded in {time.time() - t0:.1f}s")
    send({"ready": True, "sample_rate": sr})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            lang = req.get("language") or "ro"
            text = normalise(req["text"], lang)
            kw = {"text": text, "language": lang, "speed": float(req.get("speed") or 1.0)}
            if req.get("ref_audio"):
                kw["ref_audio"], kw["ref_text"] = req["ref_audio"], normalise(req["ref_text"], lang)
            elif req.get("instruct"):
                kw["instruct"] = req["instruct"]
            seed = int(req.get("seed") or 0)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            wav = model.generate(**kw)[0]
            wav = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else wav
            sf.write(req["out"], wav.reshape(-1), sr, subtype="PCM_16")
            torch.cuda.empty_cache()  # it shares the card with Breeze and Whisper
            send({"ok": True, "sample_rate": sr, "duration_s": round(len(wav.reshape(-1)) / sr, 2), "text": text})
        except Exception as exc:  # report and keep serving
            log(traceback.format_exc())
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
