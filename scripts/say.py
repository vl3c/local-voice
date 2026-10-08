"""Command-line front end for the local-voice server (standard library only).

The server must be running (start.cmd). Everything stays on 127.0.0.1.

    python scripts/say.py "We are the dark."              # defaults: reaper-drone, strength 6, seed 74525861, light room
    python scripts/say.py "We are the dark." --seed random --reverb none --no-check
    python scripts/say.py "..." --instruction "An ancient, colossal voice..." --seed 7
    python scripts/say.py "..." --preset harbinger --check --out some/dir
    python scripts/say.py "..." --like 20260101-120000-ab12                     # same voice as that take
    python scripts/say.py "Noi suntem întunericul." --lang ro          # clones the default voice
    python scripts/say.py "Noi suntem întunericul." --lang ro --like 20260101-120000-ab12
    python scripts/say.py --default-voice                               # show it
    python scripts/say.py --default-voice 20260101-120000-ab12          # set it
    python scripts/say.py --default-voice previous                      # undo the last change
    python scripts/say.py --settings                                    # the defaults in effect
    python scripts/say.py --fx 20260101-120000-ab12 --preset harbinger --check  # re-run effects only
    python scripts/say.py --fx 20260101-120000-ab12 --blend 0.5 --double 20     # half machine, half human
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
import subprocess
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


def call(path: str, payload: dict | None = None, timeout: float = 900, method: str | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"},
                                 method=method)
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


def play(path: Path) -> None:
    """Play a WAV on this machine's default output device, blocking until it ends."""
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return
    players = [["afplay"]] if sys.platform == "darwin" else [
        ["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]]
    for cmd in players:
        if shutil.which(cmd[0]):
            subprocess.run([*cmd, str(path)], check=True)
            return
    raise RuntimeError("no audio player found (tried " + ", ".join(c[0] for c in players) + ")")


def deliver(res: dict, out: Path | None, name: str | None, formats: str = "both") -> list[Path]:
    """Copy the processed audio (WAV and/or MP3), its aligned voice stem and the dry WAV to `out`,
    never overwriting."""
    paths = []
    for kind, fname in res.get("files", {}).items():
        src = OUTPUT / fname
        if out is None:
            paths.append(src)
            continue
        if kind == "spectrogram":
            paths.append(src)  # review material, not a deliverable
            continue
        if (kind in ("wet", "dry", "voice") and formats == "mp3") or (kind == "mp3" and formats == "wav"):
            continue  # mp3: just the MP3 (for messengers); wav: the WAVs (processed, voice, dry)
        out.mkdir(parents=True, exist_ok=True)
        stem = (name + {"dry": "_dry", "voice": "_voice"}.get(kind, "")) if name else Path(fname).stem
        dest = out / f"{stem}{src.suffix}"
        i = 1
        while dest.exists():
            dest = out / f"{stem}_{i}{src.suffix}"
            i += 1
        shutil.copyfile(src, dest)
        paths.append(dest)
    return paths


def given(d: dict) -> dict:
    """Drop options left unset, so the server's own defaults apply."""
    return {k: v for k, v in d.items() if v is not None}


def main() -> int:
    # Piped (as agents run it), Windows defaults stdout to cp1252, which can't print
    # Romanian letters such as ș and ț. A console already gets Unicode.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="what to say (English; (sigh)-style vocal events allowed)")
    ap.add_argument("--instruction", help="voice description. English: default the preset's suggested one ('' for "
                    "none; with --like, default is a pure clone). Romanian: attributes; default clones the default voice")
    ap.add_argument("--preset", help="effects preset (default: reaper-drone)")
    ap.add_argument("--lang", default="en", choices=["en", "ro"], help="en (default): Breeze TTS 2; ro: OmniVoice")
    ap.add_argument("--speed", type=float, help="Romanian only: the model's speaking rate, 0.75-1.3, default 0.9 "
                    "(for slower than that, use --tempo)")
    ap.add_argument("--pitch", type=float, help="semitones, -12..12, applied with the effects "
                    "(e.g. -3 for a lower, bigger voice); works with --fx too")
    ap.add_argument("--reverb", choices=["none", "room", "hall", "cathedral"],
                    help="space added last: none, room (light; the default), hall (medium), cathedral (heavy); "
                    "works with --fx too")
    ap.add_argument("--tempo", type=float, help="time-stretch, 0.5-1.5, applied with the effects "
                    "(0.85 = slower, same pitch); works with --fx too")
    ap.add_argument("--blend", type=float, help="0-1: mix the human voice under the machine (0 = all machine, "
                    "0.5 = half and half, 1 = all voice), sample-aligned; works with --fx too")
    ap.add_argument("--double", type=float, metavar="MS", help="with --blend: let the machine trail the voice by "
                    "MS ms (0-60; 10-30 thickens it into one doubled voice); works with --fx too")
    ap.add_argument("--seed", help="a number to reproduce a take; 'random' for a new voice "
                    "(default: 74525861)")
    ap.add_argument("--cfg", type=float, help="English only: instruction strength (default 6; Breeze recommends 4)")
    ap.add_argument("--like", metavar="TAKE", help="keep the voice of an earlier take (voice direction)")
    ap.add_argument("--check", action="store_true", default=None,
                    help="transcribe with Whisper and draw a spectrogram (on by default)")
    ap.add_argument("--no-check", dest="check", action="store_false", help="skip the transcription check")
    ap.add_argument("--fx", metavar="TAKE", help="re-run effects on an earlier take; no new generation")
    ap.add_argument("--presets", action="store_true", help="list presets and exit")
    ap.add_argument("--default-voice", dest="default_voice", nargs="?", const="", metavar="TAKE",
                    help="show the default voice (what Romanian clones unless told otherwise), set it to TAKE, "
                    "or 'previous' to undo the last change")
    ap.add_argument("--settings", action="store_true",
                    help="show the defaults in effect: built-in, or the user's saved ones (config.local.json)")
    ap.add_argument("--out", type=Path, help="copy results here (never overwrites)")
    ap.add_argument("--name", help="file name stem for --out")
    ap.add_argument("--format", choices=["both", "wav", "mp3"], default="both",
                    help="what --out copies: both (default: processed WAV + MP3, the aligned voice WAV and the dry "
                    "WAV), wav (the same without the MP3) or mp3 (just the MP3, e.g. for WhatsApp). --out creates "
                    "the folder if needed")
    ap.add_argument("--wait", type=float, default=5, help="seconds to wait for the server to come up")
    ap.add_argument("--json", action="store_true", help="print the raw server response")
    ap.add_argument("--no-save", action="store_true",
                    help="delete the take's files from output/ when done (after --play and/or --out): for speech "
                    "that's only meant to be heard")
    ap.add_argument("--play", action="store_true",
                    help="play the processed audio on this machine's speakers when it's ready (blocks until done)")
    args = ap.parse_args()

    if args.seed is not None and args.seed.lower() != "random" and not args.seed.isdigit():
        ap.error("--seed takes a number or 'random'")
    if not (args.text or args.fx or args.presets or args.default_voice is not None or args.settings):
        ap.error("give the text to say, or --fx TAKE, --presets, --settings or --default-voice")
    if not wait_for_server(args.wait):
        print(f"No server at {SERVER}. Start it with start.cmd (or start.cmd --no-browser).")
        return 1

    if args.presets:
        for k, v in call("/api/presets").items():
            print(f"{k}\n  {v['description']}\n  voice: {v['instruction'] or '(none)'}\n")
        return 0
    if args.settings:
        st = call("/api/settings")
        src = f"your saved defaults ({st['file']}) over the built-in ones" if st["file"] else "the built-in defaults"
        print(f"in effect: {src}")
        for k, v in st["effective"].items():
            mark = "  (saved)" if k in st["saved"] else ""
            shown = ("random" if (k == "seed" and v is None) else
                     "the preset's (English; Romanian clones the default voice)" if (k == "instruction" and not v) else v)
            print(f"  {k:12s} {shown}{mark}")
        return 0
    if args.default_voice is not None:
        if args.default_voice.lower() == "previous":
            h = call("/api/default-voice/restore-previous", {})
        else:
            h = call("/api/default-voice", {"id": args.default_voice} if args.default_voice else None)
        if not h.get("id"):
            print("no default voice yet: the first Romanian line without --instruction or --like makes one")
        else:
            print(f"default voice: take {h['id']} ({h['f0_hz']} Hz median pitch)")
            if h.get("previous"):
                p = h["previous"]
                print(f"previous: take {p['id']} ({p.get('f0_hz')} Hz); restore it with --default-voice previous")
        return 0
    if args.fx:
        res = call("/api/fx", given({"id": args.fx, "preset": args.preset, "check": args.check,
                                     "pitch": args.pitch, "tempo": args.tempo, "reverb": args.reverb,
                                     "blend": args.blend, "double_ms": args.double}))
    else:
        body = given({"text": args.text, "instruction": args.instruction, "preset": args.preset,
                      "cfg_scale": args.cfg, "like": args.like,
                      "language": args.lang, "speed": args.speed,
                      "pitch": args.pitch, "tempo": args.tempo, "reverb": args.reverb,
                      "blend": args.blend, "double_ms": args.double, "check": args.check})
        if args.seed is not None:  # omitted: the server's default seed
            body["seed"] = None if args.seed.lower() == "random" else int(args.seed)
        res = call("/api/speak", body)

    paths = deliver(res, args.out, args.name, args.format)
    if args.json:  # machine-readable: JSON only, with the delivered paths inside
        print(json.dumps({**res, "paths": [str(p) for p in paths]}, indent=2, ensure_ascii=False))
    else:
        if "id" in res:
            shape = "".join(f"  {k} {res[k]:+g}" if k == "pitch" else f"  {k} {res[k]:g}"
                            for k, neutral in (("pitch", 0), ("tempo", 1), ("blend", 0), ("double_ms", 0))
                            if res.get(k, neutral) != neutral)
            if res.get("reverb", "none") != "none":
                shape += f"  reverb {res['reverb']}"
            print(f"take {res['id']}  seed {res.get('seed')}  preset {res.get('preset')}{shape}")
            if res.get("instruction"):
                print(f"voice: {res['instruction']}")
            elif res.get("like"):
                print(f"voice: {'default voice, ' if res.get('default_voice') else ''}cloned from take {res['like']}")
        keys = ("duration_s", "fx_s") if args.fx else ("duration_s", "gen_s", "fx_s")
        timing = [f"{k} {res[k]}s" for k in keys if k in res]
        print("  ".join(timing) + ("  (effects only, no regeneration)" if args.fx else ""))
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
    status = 0
    try:
        if args.play:
            wet = res.get("files", {}).get("wet")
            if not wet:
                print("nothing to play", file=sys.stderr)
                status = 1
            else:
                try:
                    sys.stdout.flush()
                    play(OUTPUT / wet)
                    if not args.json:
                        print(f"played {wet}")
                except Exception as exc:  # the take exists; only the playback failed
                    print(f"could not play {wet}: {exc}", file=sys.stderr)
                    status = 1
    finally:
        if args.no_save and res.get("id"):
            forget(res, fx_only=bool(args.fx), quiet=args.json)
    return status


def forget(res: dict, fx_only: bool, quiet: bool) -> None:
    """--no-save: delete what this call made. For a new take that's the whole take; for
    --fx only the files it rendered, never the take being previewed."""
    if fx_only:
        names = [n for k, n in res.get("files", {}).items() if k != "dry"]
        for n in names:
            call(f"/files/{n}", method="DELETE")
        msg = f"not saved: removed this preview's {len(names)} file(s); take {res['id']} is kept"
    else:
        gone = call(f"/api/takes/{res['id']}", method="DELETE")
        msg = f"not saved: removed {gone['deleted']} file(s) of take {res['id']}"
    if not quiet:
        print(msg)


if __name__ == "__main__":
    sys.exit(main())
