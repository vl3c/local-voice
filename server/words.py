"""Word timings: when each word of the script is spoken, on a render's own timeline.

Whisper gives word times for what it heard; this maps them onto the words of the
script (so a misheard or unheard word still gets a time, flagged approximate),
snaps each word's edges to where the sound actually starts and stops, and groups
the words into phrases for subtitles.

Pure functions on numbers and strings: the server runs Whisper and passes its output in.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

# Romanian is written with both comma-below (ș ț, correct) and cedilla (ş ţ, legacy)
# letters; Whisper uses either. Compare them as equal.
FOLD = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "ș", "Ţ": "ț"})
WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
EVENT = re.compile(r"\([^)]*\)")  # vocal events in the script, like (sigh): not words
PUNCT_BREAK = re.compile(r"[.,;:!?…—–]")

FRAME_S = 0.01
ACTIVE_DB = 35.0      # a frame is sound if within this many dB of the loudest frame
MIN_SNAP_FRAMES = 8   # how far (in frames) an edge may move, at least
PAUSE_S = 0.2         # a silence this long is between words, never inside one
PHRASE_GAP_S = 0.3    # a silence this long starts a new subtitle
PHRASE_MAX_WORDS = 12


def tokens(s: str) -> list[str]:
    """Lower-case words, vocal events like (sigh) dropped: how texts are compared."""
    return WORD.findall(EVENT.sub(" ", s.lower().translate(FOLD)))


def script_words(text: str) -> list[dict]:
    """The script's words in order: {"word" as written, "norm", "pos": (start, end) in text}."""
    masked = EVENT.sub(lambda m: " " * len(m.group()), text)
    folded = masked.lower().translate(FOLD)
    if len(folded) != len(text):  # a rare letter whose lower case is longer: no positions
        return [{"word": w, "norm": w, "pos": None} for w in tokens(text)]
    return [{"word": text[m.start():m.end()], "norm": m.group(), "pos": (m.start(), m.end())}
            for m in WORD.finditer(folded)]


def heard_words(chunks: list[dict], normalise=lambda s: s) -> list[tuple[str, float, float]]:
    """Whisper's word chunks ({"text", "timestamp": (start, end)}) as (word, start, end).

    `normalise` spells numbers out the way the script does; a chunk that becomes
    several words ("2026" -> "two thousand twenty six") shares its time evenly.
    """
    out = []
    for c in chunks:
        start, end = c["timestamp"]
        if start is None:
            continue
        end = start + 0.3 if end is None or end < start else end
        ws = tokens(normalise(c["text"]))
        step = (end - start) / max(1, len(ws))
        out += [(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(ws)]
    return out


def align(script: list[dict], heard: list[tuple[str, float, float]]) -> tuple[list[dict], list[dict]]:
    """Give every script word a start and end from what Whisper heard.

    Returns (words, extras). words: one {"word", "start", "end"} per script word,
    plus "approx": True where it wasn't heard as such (misheard, merged or missing:
    its time is shared out or interpolated from its neighbours). extras: what was
    heard that isn't in the script ({"word", "start", "end"}).
    """
    a, b = [w["norm"] for w in script], [h[0] for h in heard]
    times: list[tuple[float, float] | None] = [None] * len(a)
    approx = [False] * len(a)
    extras = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = heard[j1 + k][1:]
        elif tag == "replace":
            # Misheard: share the heard span out over the script words, in order.
            s, e = heard[j1][1], heard[j2 - 1][2]
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (s + (e - s) * k / n, s + (e - s) * (k + 1) / n)
                approx[i1 + k] = True
        elif tag == "insert":
            extras += [{"word": h[0], "start": h[1], "end": h[2]} for h in heard[j1:j2]]
        # "delete": script words with nothing heard: interpolated below.
    _interpolate(times, approx)
    words = []
    for w, t, ap in zip(script, times, approx):
        if t is None:
            continue
        d = {"word": w["word"], "start": round(t[0], 3), "end": round(t[1], 3)}
        if ap:
            d["approx"] = True
        words.append(d)
    return words, extras


def _interpolate(times: list, approx: list) -> None:
    """Fill runs of unheard words evenly between the heard words around them."""
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        before = times[i - 1][1] if i > 0 else None
        after = times[j][0] if j < len(times) else None
        if before is None and after is None:
            return  # nothing heard at all: no times to give
        lo = before if before is not None else max(0.0, after - 0.3 * (j - i))
        hi = after if after is not None else lo + 0.3 * (j - i)
        n = j - i
        for k in range(n):
            times[i + k] = (lo + (hi - lo) * k / n, lo + (hi - lo) * (k + 1) / n)
            approx[i + k] = True
        i = j


def activity(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per 10 ms frame: is there sound (within ACTIVE_DB of the loudest frame)?"""
    hop = max(1, int(FRAME_S * sr))
    n = len(audio) // hop
    if not n:
        return np.zeros(0, dtype=bool)
    level = 10 * np.log10(np.mean(audio[: n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1) + 1e-12)
    return level > level.max() - ACTIVE_DB


def split_at_pauses(words: list[dict], active: np.ndarray) -> list[dict]:
    """Put the boundary between two neighbouring words in the pause between them.

    Whisper places word boundaries well in running speech but loosely around
    pauses: it can end a word early and start the next one inside the first one's
    sound, carrying the pause along. A silence of PAUSE_S or more never sits inside
    a word, so where one lies within a pair's span, the first word ends where it
    begins and the second starts where it ends (the longest such silence wins).
    """
    out = [dict(w) for w in words]
    n, need = len(active), int(round(PAUSE_S / FRAME_S))
    for a, b in zip(out, out[1:]):
        lo, hi = int(a["start"] / FRAME_S), min(n, int(np.ceil(b["end"] / FRAME_S)))
        best, run_start = None, None
        for k in range(lo, hi + 1):
            silent = k < hi and not active[k]
            if silent and run_start is None:
                run_start = k
            elif not silent and run_start is not None:
                if k - run_start >= need and (best is None or k - run_start > best[1] - best[0]):
                    best = (run_start, k)
                run_start = None
        if best and best[0] > lo and best[1] < hi:  # sound on both sides: a pause between them
            a["end"], b["start"] = round(best[0] * FRAME_S, 3), round(best[1] * FRAME_S, 3)
    return out


def snap(words: list[dict], active: np.ndarray) -> list[dict]:
    """Move each word's edges onto where its sound starts and stops.

    Whisper's word times are close but loose: a word followed by a pause often
    "ends" when the next one starts. An edge in silence moves inwards to the
    word's own sound (it can't overshoot: it stops at the word's other edge). An
    edge inside sound moves outwards to where that sound begins or ends, but never
    across a neighbour and at most half the word's length (or 80 ms): connected
    speech has no silence between words, and the next word's sound isn't this one's.
    """
    out = [dict(w) for w in words]
    n = len(active)
    if not n:
        return out
    fr = lambda t: int(np.clip(round(t / FRAME_S), 0, n - 1))  # noqa: E731
    for i, w in enumerate(out):
        s, e = fr(w["start"]), max(fr(w["start"]), fr(w["end"]) - 1)
        reach = max(MIN_SNAP_FRAMES, (e - s) // 2)
        lo = fr(out[i - 1]["end"]) if i else 0
        hi = fr(words[i + 1]["start"]) - 1 if i + 1 < len(out) else n - 1
        if active[s]:
            while s - 1 >= max(lo, s - reach) and active[s - 1]:
                s -= 1
        else:
            k = s
            while k < e and not active[k]:
                k += 1
            s = k if active[k] else s
        if active[e]:
            while e + 1 <= min(hi, e + reach) and active[e + 1]:
                e += 1
        else:
            k = e
            while k > s and not active[k]:
                k -= 1
            e = k if active[k] else e
        if e >= s:
            w["start"], w["end"] = round(s * FRAME_S, 3), round((e + 1) * FRAME_S, 3)
    return out


def phrases(words: list[dict], script: list[dict], text: str) -> list[dict]:
    """Group timed words into subtitle phrases: break at punctuation, at a pause of
    PHRASE_GAP_S or more, or after PHRASE_MAX_WORDS words. {"start", "end", "text"}."""
    by_word = [w for w in script]  # script words, same order as `words` when all are timed
    if len(by_word) != len(words):
        by_word = [None] * len(words)
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = nxt is not None and nxt["start"] - w["end"] >= PHRASE_GAP_S
        punct = False
        if nxt is not None and by_word[i] and by_word[i + 1] and by_word[i]["pos"] and by_word[i + 1]["pos"]:
            punct = bool(PUNCT_BREAK.search(text[by_word[i]["pos"][1]:by_word[i + 1]["pos"][0]]))
        if nxt is None or gap or punct or len(cur) >= PHRASE_MAX_WORDS:
            first, last = by_word[cur[0]], by_word[cur[-1]]
            if first and last and first["pos"] and last["pos"]:
                end = last["pos"][1]
                m = re.match(r"[.,;:!?…]+", text[end:])  # keep the phrase's own punctuation
                said = text[first["pos"][0]:end + (m.end() if m else 0)]
            else:
                said = " ".join(words[k]["word"] for k in cur)
            out.append({"start": words[cur[0]]["start"], "end": words[cur[-1]]["end"], "text": " ".join(said.split())})
            cur = []
    return out


def srt(cues: list[dict]) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    return "".join(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}\n\n" for i, c in enumerate(cues, 1))


STRAY_MIN_S = 0.15    # this much sound beyond the script counts as a stray vocalization
STRAY_MARGIN_S = 0.12  # ignore sound this close to the first/last word: its own breath and decay
TRIM_PAD_S = 0.15      # a trim keeps at most this much after the last word


def stray_sound(words: list[dict], active: np.ndarray) -> dict:
    """Sound outside the script: before the first word or after the last.

    {"before": {...} | None, "after": {...} | None}, each {"start", "end", "seconds"};
    "after" also has "cut": where to end the take to drop it (in the gap after the
    last word, at most TRIM_PAD_S past it). Whisper often doesn't write a hum or an
    "mmm" down at all, so this looks at the sound itself, not the transcript.
    """
    out = {"before": None, "after": None}
    if not words or not len(active):
        return out
    n, need = len(active), int(round(STRAY_MIN_S / FRAME_S))
    first = int((words[0]["start"] - STRAY_MARGIN_S) / FRAME_S)
    last = int(np.ceil((words[-1]["end"] + STRAY_MARGIN_S) / FRAME_S))
    if first > 0 and active[:first].sum() >= need:
        k = np.flatnonzero(active[:first])
        out["before"] = _span(k[0], k[-1], active[:first].sum())
    if last < n and active[last:].sum() >= need:
        k = last + np.flatnonzero(active[last:])
        span = _span(k[0], k[-1], active[last:].sum())
        gap = span["start"] - words[-1]["end"]
        span["cut"] = round(words[-1]["end"] + min(TRIM_PAD_S, gap / 2), 3)
        out["after"] = span
    return out


def voiced_share(audio: np.ndarray, sr: int, start: float, end: float) -> float:
    """How much of [start, end] has a pitch: a hum or an "mmm" is voiced, a breath isn't."""
    import parselmouth
    seg = audio[int(start * sr):int(end * sr)].astype(np.float64)
    if len(seg) < int(0.05 * sr):
        return 0.0
    f = parselmouth.Sound(seg, sampling_frequency=sr).to_pitch(
        time_step=0.01, pitch_floor=40, pitch_ceiling=500).selected_array["frequency"]
    return float(np.mean(f > 0)) if len(f) else 0.0


def _span(k0: int, k1: int, frames: int) -> dict:
    return {"start": round(k0 * FRAME_S, 3), "end": round((k1 + 1) * FRAME_S, 3), "seconds": round(frames * FRAME_S, 2)}
