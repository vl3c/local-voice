"""Command-line front end for the local-voice server (standard library only).

The server must be running (start.cmd). Everything stays on 127.0.0.1.

    python scripts/say.py "We are the dark."                          # the default: the Reaper (ancient-male, reaper-drone)
    python scripts/say.py "We are the dark." --voice ancient-female --blend 0.4 --out D:/clips
    python scripts/say.py "Come closer." --voice vampire-female --preset dry --reverb none
    python scripts/say.py "Noi suntem întunericul." --lang ro --voice young-woman
    python scripts/say.py "..." --instruction "A tired old robot, slow and kind." --seed random
    python scripts/say.py "..." --like 20260101-120000-ab12           # the voice of an earlier take
    python scripts/say.py --fx 20260101-120000-ab12 --preset wraith --reverb room   # new effects, same take
    python scripts/say.py --voices                                    # also --presets, --settings
    python scripts/say.py --unload                                    # free the GPU, keep the server

With the check (on by default) Whisper transcribes the take: a words-heard score says whether
the effects buried the speech, and --out also delivers word timings (_words.json) and subtitles
(.srt). Every option is grouped below; the server's README has the full reference.

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


def deliver(res: dict, out: Path | None, name: str | None, formats: str = "all") -> list[Path]:
    """Copy the processed audio (WAV and/or MP3), its aligned voice stem, the dry WAV and (with
    the check) the word timings and subtitles to `out`, never overwriting."""
    paths = []
    for kind, fname in res.get("files", {}).items():
        src = OUTPUT / fname
        if out is None:
            paths.append(src)
            continue
        if kind == "spectrogram":
            paths.append(src)  # review material, not a deliverable
            continue
        if (kind != "mp3" and formats == "mp3") or (kind == "mp3" and formats == "wav"):
            continue  # mp3: just the MP3 (for messengers); wav: everything else
        out.mkdir(parents=True, exist_ok=True)
        stem = (name + {"dry": "_dry", "voice": "_voice", "words": "_words"}.get(kind, "")) if name else Path(fname).stem
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


RELOAD = "~20 s, or ~1 min if the fast paths were never compiled"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="what to say ((sigh)-style vocal events allowed in English)")

    who = ap.add_argument_group("who speaks")
    who.add_argument("--voice", help="a built-in voice (--voices lists them): ancient-male (default), ancient-female, "
                     "young-woman, child, old-frail, vampire-male, vampire-female. Brings its own description and seed")
    who.add_argument("--instruction", help="your own voice description instead of --voice. English: free text "
                     "('' for none; with --like, the default is a pure clone). Romanian: attributes such as "
                     "'male, elderly, very low pitch'")
    who.add_argument("--lang", default="en", choices=["en", "ro"], help="en (default): Breeze TTS 2; ro: OmniVoice")
    who.add_argument("--ro-voice", dest="ro_voice", choices=["auto", "clone", "native"],
                     help="Romanian: speak as a clone of the voice's English take (its depth and delivery, a little "
                     "English accent), or as its native Romanian voice (native accent, higher and smaller). auto "
                     "(default): each voice's better one")
    who.add_argument("--like", metavar="TAKE", help="keep the voice of an earlier take, in either language")
    who.add_argument("--seed", help="a number to reproduce a take, or 'random' for a new voice "
                     "(default: the voice's own seed)")
    who.add_argument("--cfg", type=float, help="English: how strictly the voice follows its description "
                     "(default 6; Breeze recommends 4; 0.5-10)")
    who.add_argument("--speed", type=float, help="Romanian: the model's speaking rate, 0.75-1.3 (default 0.9); "
                     "slower than that is --tempo's job")

    fx = ap.add_argument_group("effects (all work with --fx too)")
    fx.add_argument("--preset", help="the effects: reaper-drone (default), reaper, harbinger, sovereign, demon, "
                    "synthetic-ai, ai-choir, wraith, hive, droid, dry (--presets describes them)")
    fx.add_argument("--reverb", choices=["none", "room", "hall", "cathedral"],
                    help="space added last: room (light; default), hall, cathedral, or none for a dry take")
    fx.add_argument("--pitch", type=float, help="semitones, -12..12, on the finished sound (-3: lower and bigger)")
    fx.add_argument("--tempo", type=float, help="time-stretch, 0.5-1.5, same pitch (0.85 = slower)")
    fx.add_argument("--blend", type=float, help="mix the human voice under the effect, 0-1 (0 all effect, 0.5 half "
                    "and half, 1 all human), sample-aligned")
    fx.add_argument("--bass", type=float, metavar="DB", help="shelve the bass under 150 Hz, -12..6 dB "
                    "(-6 when a take warns that the words sit low under the bass)")
    fx.add_argument("--double", type=float, metavar="MS", help="with --blend: the effect trails the voice by MS ms, "
                    "0-60 (default 0; 10-30 thickens it, and can read as extra reverb)")
    fx.add_argument("--trim-tail", dest="trim_tail", action="store_true", default=None,
                    help="cut stray sound after the last word (an 'mmm', a breath) when a take warns about it; "
                    "the take remembers the cut")
    fx.add_argument("--no-trim-tail", dest="trim_tail", action="store_false", help="undo a take's --trim-tail")

    out = ap.add_argument_group("check, output and playback")
    out.add_argument("--check", action="store_true", default=None,
                     help="transcribe with Whisper, time the words, draw a spectrogram (on by default)")
    out.add_argument("--no-check", dest="check", action="store_false", help="skip the check (~1 s faster)")
    out.add_argument("--out", type=Path, help="copy the results to this folder (created if needed; never overwrites)")
    out.add_argument("--name", help="file name stem for --out")
    out.add_argument("--format", choices=["all", "both", "wav", "mp3"], default="all",
                     help="what --out copies: all (default: the processed WAV and MP3, the aligned voice WAV, the "
                     "dry WAV, and with the check _words.json and .srt), wav (all but the MP3) or mp3 (just the "
                     "MP3: WhatsApp and most messengers reject WAV). 'both' is the old name for all")
    out.add_argument("--play", action="store_true", help="play the take on this machine's speakers (returns when done)")
    out.add_argument("--no-save", action="store_true", help="delete what this call made from output/ when done "
                     "(after --play and --out): for speech only meant to be heard")
    out.add_argument("--json", action="store_true", help="print only the server's JSON (plus the delivered paths)")

    manage = ap.add_argument_group("takes, voices and the server")
    manage.add_argument("--fx", metavar="TAKE", help="re-run effects on an earlier take: no new generation, ~1-3 s")
    manage.add_argument("--voices", action="store_true", help="list the built-in voices")
    manage.add_argument("--presets", action="store_true", help="list the effects presets")
    manage.add_argument("--settings", action="store_true",
                        help="show the defaults in effect (built-in, or the user's saved config.local.json)")
    manage.add_argument("--ro-take", "--default-voice", dest="ro_take", nargs="?", const="", metavar="TAKE",
                        help="show which stored take --voice's Romanian lines clone (with --ro-voice native: its "
                        "native one), set it to TAKE, or 'previous' to undo the last change")
    manage.add_argument("--unload", action="store_true", help="free the GPU: the server drops its models but keeps "
                        f"running and reloads them on the next request ({RELOAD}); it also does this by itself "
                        "after 15 idle minutes")
    manage.add_argument("--wait", type=float, default=5, help="seconds to wait for the server to come up (180 "
                        "after starting it)")
    return ap


def request_for(args: argparse.Namespace) -> tuple[str, dict]:
    """The API call a take needs: (path, body). Options left out stay out, so the server's
    (and the user's saved) defaults apply."""
    effects = {"preset": args.preset, "check": args.check, "pitch": args.pitch, "tempo": args.tempo,
               "reverb": args.reverb, "blend": args.blend, "double_ms": args.double, "bass_db": args.bass,
               "trim_tail": args.trim_tail}
    if args.fx:
        return "/api/fx", given({"id": args.fx, **effects})
    body = given({"text": args.text, "voice": args.voice, "instruction": args.instruction, "language": args.lang,
                  "ro_voice": args.ro_voice, "like": args.like, "cfg_scale": args.cfg, "speed": args.speed, **effects})
    if args.seed is not None:  # omitted: the voice's (or the saved) seed
        body["seed"] = None if args.seed.lower() == "random" else int(args.seed)
    return "/api/speak", body


def main() -> int:
    # Piped (as agents run it), Windows defaults stdout to cp1252, which can't print
    # Romanian letters such as ș and ț. A console already gets Unicode.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = build_parser()
    args = ap.parse_args()

    if args.seed is not None and args.seed.lower() != "random" and not args.seed.isdigit():
        ap.error("--seed takes a number or 'random'")
    if not (args.text or args.fx or args.presets or args.ro_take is not None or args.settings or args.unload
            or args.voices):
        ap.error("give the text to say, or --fx TAKE, --voices, --presets, --settings, --ro-take or --unload")
    if not wait_for_server(args.wait):
        print(f"No server at {SERVER}. Start it with start.cmd (or start.cmd --no-browser).")
        return 1

    if args.unload:
        r = call("/api/unload", {})
        print(f"unloaded {', '.join(r['unloaded']) or 'nothing (already in standby)'}; "
              f"the server now holds {r['server_reserved_gb']} GB")
        return 0
    if args.presets:
        for k, v in call("/api/presets").items():
            print(f"{k}\n  {v['description']}\n")
        return 0
    if args.voices:
        for k, v in call("/api/voices").items():
            takes = v.get("romanian_takes") or {}
            use = (v.get("romanian") or {}).get("use", "clone")
            ro = "; ".join(f"{kind}{' (auto)' if kind == use else ''}: "
                           + (f"take {t['id']}" if t else "made on first use") for kind, t in takes.items())
            print(f"{k}  ({v['name']}, seed {v['seed']})\n  {v.get('about', '')}\n  {v['instruction']}\n"
                  f"  Romanian: {ro}\n")
        return 0
    if args.settings:
        st = call("/api/settings")
        src = f"your saved defaults ({st['file']}) over the built-in ones" if st["file"] else "the built-in defaults"
        print(f"in effect: {src}")
        for k, v in st["effective"].items():
            mark = "  (saved)" if k in st["saved"] else ""
            shown = ("random" if (k == "seed" and v is None) else
                     "the voice's (Romanian clones it)" if (k == "instruction" and not v) else v)
            print(f"  {k:12s} {shown}{mark}")
        return 0
    if args.ro_take is not None:
        voice, kind = args.voice or "ancient-male", "native" if args.ro_voice == "native" else "clone"
        if args.ro_take.lower() == "previous":
            h = call("/api/default-voice/restore-previous", {"voice": voice, "kind": kind})
        elif args.ro_take:
            h = call("/api/default-voice", {"id": args.ro_take, "voice": voice, "kind": kind})
        else:
            h = call(f"/api/default-voice?voice={voice}&kind={kind}")
        what = "its native Romanian voice" if kind == "native" else "a clone of its English take"
        if not h.get("id"):
            print(f"{voice} in Romanian, as {what}: none stored yet; the first Romanian line makes it")
        else:
            print(f"{voice} in Romanian, as {what}: take {h['id']} ({h['f0_hz']} Hz median pitch)")
            if h.get("previous"):
                p = h["previous"]
                print(f"previous: take {p['id']} ({p.get('f0_hz')} Hz); restore it with --ro-take previous")
        return 0
    path, body = request_for(args)
    res = call(path, body)

    paths = deliver(res, args.out, args.name, args.format)
    if args.json:  # machine-readable: JSON only, with the delivered paths inside
        print(json.dumps({**res, "paths": [str(p) for p in paths]}, indent=2, ensure_ascii=False))
    else:
        if "id" in res:
            shape = "".join(f"  {k} {res[k]:+g}" if k == "pitch" else f"  {k} {res[k]:g}"
                            for k, neutral in (("pitch", 0), ("tempo", 1), ("blend", 0), ("double_ms", 0),
                                               ("bass_db", 0))
                            if res.get(k, neutral) != neutral)
            if res.get("trim_end_s"):
                shape += f"  cut at {res['trim_end_s']:g} s of the dry take"
            if res.get("reverb", "none") != "none":
                shape += f"  reverb {res['reverb']}"
            print(f"take {res['id']}  seed {res.get('seed')}  preset {res.get('preset')}{shape}")
            if res.get("voice") and res.get("language") == "en":
                print(f"voice: {res['voice']}")
            elif res.get("instruction"):
                print(f"voice: {res['instruction']}")
            elif res.get("default_voice"):
                how = "native Romanian" if res.get("ro_voice") == "native" else "cloned from English"
                print(f"voice: {res.get('voice') or 'default voice'}, {how} (take {res['like']})")
            elif res.get("like"):
                print(f"voice: cloned from take {res['like']}")
        keys = ("duration_s", "fx_s") if args.fx else ("duration_s", "gen_s", "fx_s")
        timing = [f"{k} {res[k]}s" for k in keys if k in res]
        print("  ".join(timing) + ("  (effects only, no regeneration)" if args.fx else ""))
        if "heard" in res:
            for k in ("dry", "wet"):
                flag = "  [Whisper looped: unreliable, listen]" if k in res.get("asr_unreliable", []) else ""
                print(f"heard ({k}, {res['match'][k]:.0%} of words): {res['heard'][k]}{flag}")
        if res.get("words"):
            rough = sum(1 for w in res["words"] if w.get("approx"))
            print(f"word timings: {len(res['words'])} words" + (f" ({rough} approximate)" if rough else "")
                  + " in _words.json, phrases in .srt")
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
