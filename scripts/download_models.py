"""Fetch the weights listed in models.json, verifying every file.

Resumable: a partial file is continued with an HTTP Range request, and anything
whose SHA256 already matches is skipped. Nothing here is committed to the repo -
the weights land in models/ (gitignored) and keep their upstream licences.

    python scripts/download_models.py                 # everything required
    python scripts/download_models.py --verify        # check, download nothing
    python scripts/download_models.py --only breeze-tts-2
    python scripts/download_models.py --reuse D:/other/local-voice/models   # no re-download

--reuse takes the models/ folder of another install: matching files are
hard-linked from it (no extra disk space; deleting either copy leaves the other
intact), or copied if it is on another drive, then checksum-verified like any
download. Anything missing or different there is downloaded as usual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB
REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path, progress: bool = True) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    seen = 0
    last = 0.0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
            seen += len(chunk)
            if progress and total > 50 * CHUNK and time.time() - last > 0.5:
                print(f"\r    hashing {100 * seen / total:5.1f}%", end="", flush=True)
                last = time.time()
    if progress and total > 50 * CHUNK:
        print("\r" + " " * 24 + "\r", end="")
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, expect_sha: str) -> bool:
    """Download `url` to `dest`, resuming if a partial file exists. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "local-voice/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"    resuming at {human(have)}")

    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:  # range not satisfiable: the file is already complete
            resp = None
        else:
            print(f"    HTTP {exc.code}: {exc.reason}")
            return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    network error: {exc}")
        print("    Check your connection and re-run; the partial file is kept and will resume.")
        return False

    if resp is not None:
        # A 200 to a Range request means the server ignored it; restart from scratch.
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0
        total = int(resp.headers.get("Content-Length", 0)) + have
        seen = have
        last = 0.0
        with dest.open(mode) as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                seen += len(chunk)
                if time.time() - last > 0.5:
                    pct = f"{100 * seen / total:5.1f}%" if total else "  ?  "
                    print(f"\r    {pct}  {human(seen)}", end="", flush=True)
                    last = time.time()
        print(f"\r    {human(seen)} downloaded" + " " * 12)

    actual = sha256(dest)
    if actual != expect_sha:
        print(f"    CHECKSUM MISMATCH\n      expected {expect_sha}\n      got      {actual}")
        # Remove it rather than leaving a bad file that a later resume would append to.
        dest.unlink(missing_ok=True)
        print("    Discarded. Re-run to download it again.")
        return False
    print("    checksum OK")
    return True


def reuse(src: Path, dest: Path, size: int) -> bool:
    """Hard-link (or copy) `src` to `dest` if it has the expected size. True if placed."""
    if not src.is_file() or src.stat().st_size != size:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        print("    hard-linked from the other install")
    except OSError:  # another drive, or a filesystem without hard links
        print("    copying from the other install...")
        shutil.copyfile(src, dest)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO), help="folder that models/ lives in")
    ap.add_argument("--only", action="append", help="just this repo name (repeatable)")
    ap.add_argument("--verify", action="store_true", help="verify existing files, download nothing")
    ap.add_argument("--reuse", metavar="MODELS_DIR",
                    help="another install's models/ folder: hard-link matching files instead of downloading")
    args = ap.parse_args()

    spec = json.loads((REPO / "models.json").read_text(encoding="utf-8"))
    root = Path(args.root).resolve()
    repos = [r for r in spec["repos"] if not args.only or r["name"] in args.only]

    total = sum(f["bytes"] for r in repos for f in r["files"])
    print(f"{len(repos)} model(s), about {human(total)}\ninto {root / 'models'}\n")

    failed = []
    for r in repos:
        print(f"== {r['name']}  ({r['source']} @ {r['revision'][:10]})")
        if r.get("note"):
            print(f"   {r['note']}")
        for f in r["files"]:
            dest = root / r["dest"] / f["path"]
            label = f"{r['name']}/{f['path']}"
            print(f"  {f['path']}  ({human(f['bytes'])})")

            if dest.exists() and dest.stat().st_size == f["bytes"]:
                if sha256(dest) == f["sha256"]:
                    print("    already present, checksum OK")
                    continue
                print("    present but checksum does NOT match")
                if args.verify:
                    failed.append(label)
                    continue
                print("    re-downloading")
                dest.unlink()
            elif args.verify:
                print("    missing")
                failed.append(label)
                continue
            elif args.reuse and reuse(Path(args.reuse) / Path(r["dest"]).name / f["path"], dest, f["bytes"]):
                if sha256(dest) == f["sha256"]:
                    print("    checksum OK")
                    continue
                print("    the other install's copy does NOT match; downloading instead")
                dest.unlink()

            if not download(f"{r['base_url']}/{f['path']}", dest, f["sha256"]):
                failed.append(label)
        print()

    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("All weights present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
