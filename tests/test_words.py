r"""Word timings: matching Whisper's words to the script, snapping, phrases, subtitles.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

import words as wt  # noqa: E402
from textnorm import normalise  # noqa: E402


def chunk(text, start, end):
    return {"text": text, "timestamp": (start, end)}


def test_script_words_keep_their_spelling_and_skip_events():
    s = wt.script_words("We are (sigh) the dark. Don't run!")
    assert [w["word"] for w in s] == ["We", "are", "the", "dark", "Don't", "run"]
    assert [w["norm"] for w in s][:2] == ["we", "are"]


def test_every_script_word_gets_a_time():
    text = "On the fifth, you open your hand."
    heard = wt.heard_words([chunk(" On", 0.0, 0.2), chunk(" the", 0.2, 0.36), chunk(" fist,", 0.36, 1.1),
                            chunk(" open", 1.66, 2.0), chunk(" your", 2.0, 2.2), chunk(" hand.", 2.2, 2.7)])
    words, extras = wt.align(wt.script_words(text), heard)
    assert [w["word"] for w in words] == ["On", "the", "fifth", "you", "open", "your", "hand"]
    assert words[2].get("approx") and words[3].get("approx")  # misheard, and not heard at all
    assert not words[0].get("approx") and words[-1]["end"] == 2.7
    assert words[3]["start"] >= words[2]["end"] and words[3]["end"] <= words[4]["start"]
    assert extras == []


def test_numbers_match_once_spelled_out():
    heard = wt.heard_words([chunk(" In", 0.0, 0.2), chunk(" 2026.", 0.3, 1.5)], lambda s: normalise(s, "en"))
    words, _ = wt.align(wt.script_words("In twenty twenty six."), heard)
    assert len(words) == 4 and words[-1]["end"] == 1.5


def test_extra_words_are_reported():
    heard = wt.heard_words([chunk(" Run.", 0.0, 0.4), chunk(" Mmm.", 0.9, 1.3)])
    words, extras = wt.align(wt.script_words("Run."), heard)
    assert [w["word"] for w in words] == ["Run"]
    assert extras == [{"word": "mmm", "start": 0.9, "end": 1.3}]


def test_snap_pulls_a_loose_end_back_to_the_sound():
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    audio[int(0.10 * sr):int(0.50 * sr)] = 0.5   # "fifth" really ends at 0.50
    audio[int(1.00 * sr):int(1.40 * sr)] = 0.5   # "you" starts at 1.00
    words = [{"word": "fifth", "start": 0.12, "end": 0.98}, {"word": "you", "start": 0.98, "end": 1.45}]
    a, b = wt.snap(words, wt.activity(audio, sr))
    assert abs(a["start"] - 0.10) <= 0.011 and abs(a["end"] - 0.50) <= 0.011
    assert abs(b["start"] - 1.00) <= 0.011 and abs(b["end"] - 1.40) <= 0.011


def test_phrases_break_at_punctuation_and_pauses():
    text = "Each was right to fear. Together, you were wrong"
    script = wt.script_words(text)
    t = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.1), (1.1, 1.5),
         (2.2, 2.9), (2.9, 3.2), (3.2, 3.4), (3.4, 3.9)]
    words = [{"word": w["word"], "start": s, "end": e} for w, (s, e) in zip(script, t)]
    cues = wt.phrases(words, script, text)
    assert [c["text"] for c in cues] == ["Each was right to fear.", "Together,", "you were wrong"]
    assert cues[0]["start"] == 0.0 and cues[0]["end"] == 1.5


def test_srt_format():
    out = wt.srt([{"start": 0.0, "end": 1.5, "text": "Each was right."}, {"start": 61.25, "end": 62.0, "text": "No."}])
    assert out == "1\n00:00:00,000 --> 00:00:01,500\nEach was right.\n\n2\n00:01:01,250 --> 00:01:02,000\nNo.\n\n"


def test_a_pause_inside_a_pair_becomes_their_boundary():
    sr = 16000
    audio = np.zeros(sr * 3, dtype=np.float32)
    audio[int(0.0 * sr):int(0.90 * sr)] = 0.5    # "Together," really ends at 0.90
    audio[int(1.50 * sr):int(1.70 * sr)] = 0.5   # "you" really starts at 1.50
    words = [{"word": "Together", "start": 0.0, "end": 0.45}, {"word": "you", "start": 0.45, "end": 1.70}]
    active = wt.activity(audio, sr)
    a, b = wt.snap(wt.split_at_pauses(words, active), active)
    assert abs(a["end"] - 0.90) <= 0.011 and abs(b["start"] - 1.50) <= 0.011


def test_stray_sound_after_the_last_word_is_found_with_a_cut_in_the_gap():
    sr = 16000
    audio = np.zeros(sr * 3, dtype=np.float32)
    audio[int(0.2 * sr):int(1.0 * sr)] = 0.5    # the line
    audio[int(1.6 * sr):int(2.2 * sr)] = 0.2    # "mmm"
    words = [{"word": "hand", "start": 0.2, "end": 1.0}]
    s = wt.stray_sound(words, wt.activity(audio, sr))
    assert s["before"] is None
    assert abs(s["after"]["start"] - 1.6) <= 0.011 and s["after"]["seconds"] >= 0.5
    assert 1.0 < s["after"]["cut"] <= 1.15


def test_a_clean_take_has_no_stray_sound():
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    audio[int(0.2 * sr):int(1.0 * sr)] = 0.5
    s = wt.stray_sound([{"word": "hand", "start": 0.2, "end": 1.0}], wt.activity(audio, sr))
    assert s == {"before": None, "after": None}
