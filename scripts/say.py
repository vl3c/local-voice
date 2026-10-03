"""Command-line front end for the local-voice server (standard library only).

The server must be running (start.cmd). Everything stays on 127.0.0.1.

    python scripts/say.py "We are the dark."                                    # reaper-drone preset
    python scripts/say.py "..." --instruction "An ancient, colossal voice..." --seed 7
    python scripts/say.py "..." --preset harbinger --check --out some/dir
    python scripts/say.py "..." --like 20260101-120000-ab12                     # same voice as that take
    python scripts/say.py "Noi suntem întunericul." --lang ro          # clones the house voice
    python scripts/say.py "Noi suntem întunericul." --lang ro --like 20260101-120000-ab12
    python scripts/say.py --house-voice                                 # show it
    python scripts/say.py --house-voice 20260101-120000-ab12            # set it
    python scripts/say.py --fx 20260101-120000-ab12 --preset harbinger --check  # re-run effects only
    python scripts/say.py --presets

--check transcribes the dry and processed audio with Whisper and draws a
spectrogram: the words-heard score says whether the effects buried the speech.

Exit status: 0 done, 1 failed or no server, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import http.client
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8190"
# Never through a proxy: urllib honours HTTP_PROXY even for 127.0.0.1 unless NO_PROXY
# says otherwise, which would hand the user's text to whatever the proxy is.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
OUTPUT = Path(__file__).resolve().parent.parent / "output"


def call(path: str, payload: dict | None = None, timeout: float = 900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail")
        except Exception:
            detail = exc.reason
        sys.exit(f"server said {exc.code}: {detail}")
    except (OSError, http.client.HTTPException) as exc:
        # Refused, reset or timed out: the server is down, restarting or was stopped mid-take.
        print(f"lost the server at {SERVER}: {exc}", file=sys.stderr)
        sys.exit(1)


def wait_for_server(seconds: float) -> bool:
    deadline = time.time() + seconds
    while True:
        try:
            _OPENER.open(SERVER + "/api/health", timeout=5)
            return True
        except OSError:
            if time.time() >= deadline:
                return False
            time.sleep(2)


def deliver(res: dict, out: Path | None, name: str | None, formats: str = "both") -> list[Path]:
    """Copy the processed audio (WAV and/or MP3) and the dry WAV to `out`, never overwriting."""
    paths = []
    for kind, fname in res.get("files", {}).items():
        src = OUTPUT / fname
        if out is None:
            paths.append(src)
            continue
        if kind == "spectrogram":
            paths.append(src)  # review material, not a deliverable
            continue
        if (kind == "wet" and formats == "mp3") or (kind == "mp3" and formats == "wav"):
            continue
        out.mkdir(parents=True, exist_ok=True)
        stem = (name + ("_dry" if kind == "dry" else "")) if name else Path(fname).stem
        dest = out / f"{stem}{src.suffix}"
        i = 1
        while dest.exists():
            dest = out / f"{stem}_{i}{src.suffix}"
            i += 1
        shutil.copyfile(src, dest)
        paths.append(dest)
    return paths


def main() -> int:
    # Piped (as agents run it), Windows defaults stdout to cp1252, which can't print
    # Romanian letters such as ș and ț. A console already gets Unicode.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="what to say (English; (sigh)-style vocal events allowed)")
    ap.add_argument("--instruction", help="voice description. English: default the preset's suggested one ('' for "
                    "none; with --like, default is a pure clone). Romanian: attributes; default clones the house voice")
    ap.add_argument("--preset", default="reaper-drone")
    ap.add_argument("--lang", default="en", choices=["en", "ro"], help="en (default): Breeze TTS 2; ro: OmniVoice")
    ap.add_argument("--speed", type=float, help="Romanian only: the model's speaking rate, 0.75-1.3, default 0.85 "
                    "(for slower than that, use --tempo)")
    ap.add_argument("--pitch", type=float, default=0.0, help="semitones, -12..12, applied with the effects "
                    "(e.g. -3 for a lower, bigger voice); works with --fx too")
    ap.add_argument("--reverb", choices=["none", "room", "hall", "cathedral"], default="none",
                    help="space added last: none (default), room (light), hall (medium), cathedral (heavy); "
                    "works with --fx too")
    ap.add_argument("--tempo", type=float, default=1.0, help="time-stretch, 0.5-1.5, applied with the effects "
                    "(0.85 = slower, same pitch); works with --fx too")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cfg", type=float, default=4.0, help="Breeze only: instruction strength (default 4)")
    ap.add_argument("--like", metavar="TAKE", help="keep the voice of an earlier take (voice direction)")
    ap.add_argument("--check", action="store_true", help="transcribe with Whisper and draw a spectrogram")
    ap.add_argument("--fx", metavar="TAKE", help="re-run effects on an earlier take; no new generation")
    ap.add_argument("--presets", action="store_true", help="list presets and exit")
    ap.add_argument("--house-voice", nargs="?", const="", metavar="TAKE",
                    help="show the house voice (what Romanian clones by default), or set it to TAKE")
    ap.add_argument("--out", type=Path, help="copy results here (never overwrites)")
    ap.add_argument("--name", help="file name stem for --out")
    ap.add_argument("--format", choices=["both", "wav", "mp3"], default="both",
                    help="processed audio to copy with --out: WAV, MP3 (for WhatsApp etc.) or both (default)")
    ap.add_argument("--wait", type=float, default=5, help="seconds to wait for the server to come up")
    ap.add_argument("--json", action="store_true", help="print the raw server response")
    args = ap.parse_args()

    if not (args.text or args.fx or args.presets or args.house_voice is not None):
        ap.error("give the text to say, or --fx TAKE, --presets or --house-voice")
    if not wait_for_server(args.wait):
        print(f"No server at {SERVER}. Start it with start.cmd (or start.cmd --no-browser).")
        return 1

    if args.presets:
        for k, v in call("/api/presets").items():
            print(f"{k}\n  {v['description']}\n  voice: {v['instruction'] or '(none)'}\n")
        return 0
    if args.house_voice is not None:
        h = call("/api/house-voice", {"id": args.house_voice} if args.house_voice else None)
        print(f"house voice: take {h['id']} ({h['f0_hz']} Hz median pitch)" if h.get("id")
              else "no house voice yet: the first Romanian line without --instruction or --like makes one")
        return 0
    if args.fx:
        res = call("/api/fx", {"id": args.fx, "preset": args.preset, "check": args.check,
                               "pitch": args.pitch, "tempo": args.tempo, "reverb": args.reverb})
    else:
        res = call("/api/speak", {"text": args.text, "instruction": args.instruction, "preset": args.preset,
                                  "seed": args.seed, "cfg_scale": args.cfg, "like": args.like,
                                  "language": args.lang, "speed": args.speed,
                                  "pitch": args.pitch, "tempo": args.tempo, "reverb": args.reverb,
                                  "check": args.check})

    paths = deliver(res, args.out, args.name, args.format)
    if args.json:  # machine-readable: JSON only, with the delivered paths inside
        print(json.dumps({**res, "paths": [str(p) for p in paths]}, indent=2, ensure_ascii=False))
        return 0
    else:
        if "id" in res:
            shape = "".join(f"  {k} {res[k]:+g}" if k == "pitch" else f"  {k} {res[k]:g}"
                            for k, neutral in (("pitch", 0), ("tempo", 1)) if res.get(k, neutral) != neutral)
            if res.get("reverb", "none") != "none":
                shape += f"  reverb {res['reverb']}"
            print(f"take {res['id']}  seed {res.get('seed')}  preset {res.get('preset')}{shape}")
            if res.get("instruction"):
                print(f"voice: {res['instruction']}")
            elif res.get("like"):
                print(f"voice: {'house voice, ' if res.get('house_voice') else ''}cloned from take {res['like']}")
        timing = [f"{k} {res[k]}s" for k in ("duration_s", "gen_s", "fx_s") if k in res]
        print("  ".join(timing))
        if "heard" in res:
            for k in ("dry", "wet"):
                flag = "  [Whisper looped: unreliable, listen]" if k in res.get("asr_unreliable", []) else ""
                print(f"heard ({k}, {res['match'][k]:.0%} of words): {res['heard'][k]}{flag}")
        if res.get("check_error"):
            print(f"check failed (the take itself is fine): {res['check_error']}")
        for w in res.get("warnings", []):
            print(f"WARNING: {w}")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
