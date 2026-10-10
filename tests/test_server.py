r"""The server with fake models: endpoints, the check, word timings, warnings, tail cuts,
Romanian generation and voices, standby. No GPU and no weights needed.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

SR = 24000


def voice_like(seconds=2.0, f0=110.0, sr=SR, start=0.2, stop=None):
    """A buzzy voice from `start` to `stop` seconds (default: the end), silence around it."""
    t = np.arange(int(sr * seconds)) / sr
    x = 0.3 * np.sign(np.sin(2 * np.pi * f0 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    gate = (t >= start) & (t < (stop if stop is not None else seconds))
    return (x * gate).astype(np.float32)


class FakeBreeze:
    """Breeze stand-in: a buzz whose pitch is chosen per seed (for the default-voice search)."""
    sample_rate = SR

    def __init__(self, f0_by_seed=None):
        self.f0_by_seed = f0_by_seed or {}
        self.calls = []

    def synth(self, text, instruction, seed, cfg_scale, ref_audio=None, ref_text=None):
        self.calls.append({"text": text, "instruction": instruction, "seed": seed, "ref": ref_audio})
        return voice_like(f0=self.f0_by_seed.get(seed, 110.0)), cfg_scale


class FakeWhisper:
    """Whisper stand-in: hears `heard` (default: the script), with evenly spread word times."""

    def __init__(self):
        self.heard = None       # None: whatever the request said
        self.words_heard = None
        self.fail_words = False
        self.script = ""

    def __call__(self, audio, sr, expected_words=60, language="en"):
        return self.heard if self.heard is not None else self.script

    def words(self, audio, sr, language="en"):
        if self.fail_words:
            raise RuntimeError("whisper broke")
        text = self.words_heard if self.words_heard is not None else self.script
        ws = text.split()
        step = 1.4 / max(1, len(ws))
        return [{"text": " " + w, "timestamp": (0.25 + i * step, 0.25 + (i + 1) * step)} for i, w in enumerate(ws)]


@pytest.fixture()
def server(tmp_path, monkeypatch):
    import app  # noqa: PLC0415
    monkeypatch.setattr(app, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(app, "VOICES", tmp_path / "voices")
    monkeypatch.setattr(app, "SETTINGS_FILE", tmp_path / "config.local.json")
    (tmp_path / "output").mkdir()
    breeze, whisper = FakeBreeze(), FakeWhisper()
    monkeypatch.setattr(app, "tts", lambda: breeze)
    monkeypatch.setattr(app, "asr", lambda: whisper)
    app._fakes = (breeze, whisper)
    return app


def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app.app, base_url="http://127.0.0.1:8190")


def make_take(app, tid, text="We are the dark.", audio=None, language="en", **meta):
    sf.write(app.take_paths(tid)["dry"], voice_like() if audio is None else audio, SR)
    app.take_paths(tid)["meta"].write_text(json.dumps({"id": tid, "text": text, "language": language, **meta}),
                                           encoding="utf-8")
    return tid


def speak(app, **body):
    app._fakes[1].script = body.get("text", "")
    return app.speak(app.SpeakReq(**{"check": False, **body}))


# ------------------------------------------------------------------ endpoints

def test_health_reports_models_and_standby(server):
    h = client(server).get("/api/health").json()
    assert h["ok"] and "standby" in h and "idle_s" in h and "vram_used_gb" in h


def test_presets_and_voices_listings(server):
    c = client(server)
    presets = c.get("/api/presets").json()
    assert "reaper-drone" in presets and set(presets["ai-choir"]) == {"description"}
    voices = c.get("/api/voices").json()
    assert "ancient-male" in voices and voices["young-woman"]["romanian"]["use"] == "native"
    assert voices["ancient-male"]["romanian_takes"] == {"clone": None, "native": None}


def test_the_page_is_served(server):
    r = client(server).get("/")
    assert r.status_code == 200 and "<title>local-voice</title>" in r.text


def test_files_stay_inside_output(server):
    c = client(server)
    tid = make_take(server, "20260101-120000-aaaa")
    assert c.get(f"/files/{tid}_dry.wav").status_code == 200
    for bad in ("..%2Fconfig.local.json", "nope.wav", "..\\app.py"):
        assert c.get(f"/files/{bad}").status_code == 404


def test_fx_endpoint_renders_with_the_new_options(server):
    tid = make_take(server, "20260101-120000-bbbb")
    r = client(server).post("/api/fx", json={"id": tid, "check": False, "preset": "reaper", "pitch": -2,
                                             "blend": 0.3, "double_ms": 20, "bass_db": -6, "reverb": "none"})
    assert r.status_code == 200
    res = r.json()
    assert res["files"]["wet"] == f"{tid}_fx-reaper_pitch-2_blend0.3_double20_bass-6.wav"
    assert (server.OUTPUT / res["files"]["voice"]).is_file() and res["channels"] == 1
    assert client(server).post("/api/fx", json={"id": "nope"}).status_code == 400


def test_a_stereo_preset_writes_two_channels(server):
    tid = make_take(server, "20260101-120000-cccc")
    res = client(server).post("/api/fx", json={"id": tid, "check": False, "preset": "ai-choir"}).json()
    wav, _ = sf.read(server.OUTPUT / res["files"]["wet"])
    mp3, _ = sf.read(server.OUTPUT / res["files"]["mp3"])
    voice, _ = sf.read(server.OUTPUT / res["files"]["voice"])
    assert res["channels"] == 2 and wav.ndim == 2 and wav.shape[1] == 2 and mp3.ndim == 2 and voice.ndim == 1


def test_fx_name_lists_every_setting():
    import app
    assert app._fx_name("reaper", 0, 1.0, "none") == "fx-reaper"
    assert app._fx_name("reaper", 3, 0.9, "hall", 0.5, 0, -6) == "fx-reaper_pitch+3_tempo0.9_blend0.5_bass-6_hall"
    assert app._fx_name("reaper", 0, 1.0, "none", 0.0, 20, 0) == "fx-reaper"  # doubling only counts in a blend


# ------------------------------------------------------------------ the check: words, timings, warnings

def test_the_check_scores_times_words_and_writes_subtitles(server):
    res = speak(server, text="We are the dark.", check=True)
    assert res["match"] == {"dry": 1.0, "wet": 1.0}
    assert [w["word"] for w in res["words"]] == ["We", "are", "the", "dark"]
    words = json.loads((server.OUTPUT / res["files"]["words"]).read_text(encoding="utf-8"))
    assert words["take"] == res["id"] and words["phrases"][0]["text"] == "We are the dark."
    assert (server.OUTPUT / res["files"]["srt"]).read_text(encoding="utf-8").startswith("1\n00:00:")
    assert (server.OUTPUT / res["files"]["spectrogram"]).is_file()


def test_a_misheard_word_gets_an_approximate_time(server):
    server._fakes[1].words_heard = "We are the bark."
    res = speak(server, text="We are the dark.", check=True)
    assert res["words"][3]["word"] == "dark" and res["words"][3].get("approx")


def test_stray_warnings_name_the_sound():
    import app
    timing = {"words": [{"word": "win", "start": 0.2, "end": 1.0}],
              "extras": [{"word": "mmm", "start": 1.6, "end": 2.0}, {"word": "uh", "start": 0.5, "end": 0.6}],
              "stray": {"after": {"start": 1.6, "end": 2.1, "seconds": 0.5, "cut": 1.15, "voiced": 0.8},
                        "before": {"start": 0.0, "end": 0.1, "seconds": 0.1, "voiced": 0.0}}}
    w = app.stray_warnings(timing, trimmed=False)
    assert any("voiced sound" in x and 'heard as "mmm"' in x and "--trim-tail" in x for x in w)
    assert any("breath or noise before the first word" in x for x in w)
    assert any('"uh" at 0.50 s' in x for x in w)
    assert not any("after the last word" in x for x in app.stray_warnings(timing, trimmed=True))


def test_word_timing_finds_a_stray_hum(server):
    import voicefx
    audio = voice_like(seconds=3.0, start=0.2, stop=1.4) + voice_like(seconds=3.0, f0=180, start=2.0, stop=2.6) * 0.5
    timeline = voicefx.timed_voice(audio, SR, voicefx.load_preset("dry"))
    server._fakes[1].words_heard = "We are the dark."
    timing = server.word_timing("We are the dark.", timeline, "en")
    assert timing["stray"]["after"] and timing["stray"]["after"]["voiced"] > 0.5


def test_speech_band_warnings(server, monkeypatch):
    tid = make_take(server, "20260101-120000-dddd")
    for level, expect in ((-30.0, "buried under the bass"), (-26.0, "sit low under the bass"), (-22.0, None)):
        monkeypatch.setattr(server.voicefx, "speech_lufs", lambda y, v=level: v)
        res = client(server).post("/api/fx", json={"id": tid, "check": False}).json()
        found = [w for w in res.get("warnings", []) if "bass" in w]
        assert (expect in found[0]) if expect else not found


def test_trim_tail_cuts_remembers_and_undoes(server):
    hum = voice_like(seconds=3.0, start=0.2, stop=1.4) + voice_like(seconds=3.0, f0=180, start=2.0, stop=2.6) * 0.5
    tid = make_take(server, "20260101-120000-eeee", audio=hum)
    server._fakes[1].script = server._fakes[1].words_heard = "We are the dark."
    c = client(server)
    full = c.post("/api/fx", json={"id": tid, "check": False, "preset": "dry", "reverb": "none"}).json()
    cut = c.post("/api/fx", json={"id": tid, "check": False, "preset": "dry", "reverb": "none", "trim_tail": True}).json()
    assert cut["trim_end_s"] and 1.4 <= cut["trim_end_s"] <= 2.0 and cut["duration_s"] < full["duration_s"]
    again = c.post("/api/fx", json={"id": tid, "check": False, "preset": "dry", "reverb": "none"}).json()
    assert again["trim_end_s"] == cut["trim_end_s"]  # the take remembers
    undone = c.post("/api/fx", json={"id": tid, "check": False, "preset": "dry", "reverb": "none",
                                     "trim_tail": False}).json()
    assert undone["trim_end_s"] is None and undone["duration_s"] == full["duration_s"]


def test_trim_tail_survives_a_whisper_failure(server):
    tid = make_take(server, "20260101-120000-ffff")
    server._fakes[1].fail_words = True
    res = client(server).post("/api/fx", json={"id": tid, "check": False, "trim_tail": True}).json()
    assert res["trim_end_s"] is None and any("--trim-tail failed" in w for w in res["warnings"])


def test_nothing_to_trim_says_so(server):
    tid = make_take(server, "20260101-120000-abab", audio=voice_like(stop=1.6))
    server._fakes[1].script = "We are the dark."
    res = client(server).post("/api/fx", json={"id": tid, "check": False, "trim_tail": True}).json()
    assert any("nothing after the last word" in w for w in res["warnings"])


# ------------------------------------------------------------------ generation

def test_the_breeze_ceiling_warns(server, monkeypatch):
    monkeypatch.setattr(server, "MAX_TAKE_S", 1.5)
    res = speak(server, text="A long line.")
    assert any("ceiling" in w for w in res["warnings"])


def test_a_model_error_is_a_clear_500(server):
    from fastapi import HTTPException
    server._fakes[0].synth = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    with pytest.raises(HTTPException) as e:
        speak(server, text="Hello.")
    assert e.value.status_code == 500 and "CUDA out of memory" in e.value.detail


def test_split_sentences_merges_short_ones():
    import app
    assert app.split_sentences("Da. Nu știu ce să zic acum. Bine, atunci plecăm împreună.") == [
        "Da. Nu știu ce să zic acum.", "Bine, atunci plecăm împreună."]
    assert app.split_sentences("O singură propoziție") == ["O singură propoziție"]


class FakeOmni:
    """OmniVoice stand-in: writes a buzz and reports the text it was given (as the worker does:
    a dropped word only shows in what Whisper hears)."""

    def __init__(self):
        self.calls = []

    def synth(self, text, language, instruct, seed, speed, out, ref_audio=None, ref_text=None):
        self.calls.append(seed)
        sf.write(out, voice_like(f0=150), SR)
        return {"ok": True, "text": text}


def test_romanian_retries_a_sentence_with_missing_words(server, monkeypatch):
    engine = FakeOmni()
    monkeypatch.setattr(server, "omni", lambda: engine)
    heard = iter(["Am așteptat foarte mult", "Am așteptat foarte mult timp."])  # 1st try drops a word
    monkeypatch.setattr(server, "asr", lambda: lambda audio, sr, n, lang: next(heard))
    r = server.omni_take("Am așteptat foarte mult timp.", "ro", None, 7, 0.9, server.OUTPUT / "t_dry.wav", None, None)
    assert len(engine.calls) == 2 and r["sentences"][0]["tries"] == 2 and r["sentences"][0]["match"] == 1.0
    assert (server.OUTPUT / "t_dry.wav").is_file() and not (server.OUTPUT / "t_dry_part.wav").exists()


def test_romanian_keeps_the_best_try_and_warns(server, monkeypatch):
    monkeypatch.setattr(server, "omni", lambda: FakeOmni())
    monkeypatch.setattr(server, "asr", lambda: lambda audio, sr, n, lang: "Am așteptat")
    ref = make_take(server, "20260101-120000-acac")
    monkeypatch.setattr(server, "default_voice_ref", lambda v, k="clone": (server.take_paths(ref)["dry"], "x", ref))
    res = speak(server, text="Am așteptat foarte mult timp.", language="ro")
    assert res["sentences"][0]["tries"] == server.OMNI_TRIES
    assert any("missing words" in w for w in res["warnings"])


# ------------------------------------------------------------------ voices for Romanian

def test_the_default_voice_search_keeps_the_deepest_under_75_hz(server):
    server._fakes[0].f0_by_seed = {12: 95.0, 7: 66.0, 21: 60.0}
    wav, text, tid = server.default_voice_ref()
    info = server.default_voice_info()
    assert info["id"] == tid and 60 <= info["f0_hz"] <= 72  # seed 7 (66 Hz) wins and the search stops
    assert [c["seed"] for c in server._fakes[0].calls] == [12, 7]
    assert len(list(server.OUTPUT.glob("*_dry.wav"))) == 1  # the losing candidate was deleted


def test_another_voice_clone_is_one_take_from_its_own_description(server):
    wav, text, tid = server.default_voice_ref("child")
    call = server._fakes[0].calls[-1]
    v = server.voice_catalog()["child"]
    assert call["instruction"] == v["instruction"] and call["seed"] == v["seed"] and text == server.VOICE_TEXT
    assert server.default_voice_info("child", "clone")["id"] == tid


def test_default_voice_endpoints_per_voice_and_kind(server):
    c = client(server)
    a = make_take(server, "20260101-120000-a1a1")
    b = make_take(server, "20260101-120000-b2b2", language="ro")
    assert c.get("/api/default-voice?voice=child").json() == {"id": None, "voice": "child", "kind": "clone"}
    assert c.post("/api/default-voice", json={"id": a, "voice": "child"}).json()["id"] == a
    assert c.post("/api/default-voice", json={"id": b, "voice": "child", "kind": "native"}).json()["kind"] == "native"
    assert c.get("/api/default-voice?voice=child&kind=native").json()["id"] == b
    assert c.get("/api/default-voice?voice=child").json()["id"] == a  # the clone is untouched
    c.post("/api/default-voice", json={"id": b, "voice": "child"})
    assert c.post("/api/default-voice/restore-previous", json={"voice": "child"}).json()["id"] == a
    assert c.get("/api/default-voice?voice=nobody").status_code == 400
    assert c.post("/api/default-voice/restore-previous", json={"voice": "old-frail"}).status_code == 404


# ------------------------------------------------------------------ standby

def test_unload_drops_the_models(server, monkeypatch):
    monkeypatch.setattr(server, "_tts", object())
    monkeypatch.setattr(server, "_asr", object())
    r = client(server).post("/api/unload").json()
    assert set(r["unloaded"]) == {"tts", "whisper"} and server._tts is None and server._asr is None
    assert client(server).post("/api/unload").json()["unloaded"] == []


def test_standby_waits_for_idle_and_for_a_free_lock(server, monkeypatch):
    monkeypatch.setattr(server, "_tts", object())
    server._touch()
    assert not server._standby_tick(1)                   # used just now
    monkeypatch.setattr(server, "_last_used", time.monotonic() - 120)
    with server._lock:
        assert not server._standby_tick(1)               # a take is in progress
    assert server._standby_tick(1) and server._tts is None
    assert not server._standby_tick(1)                   # nothing left to unload


def test_a_take_resets_the_idle_clock(server, monkeypatch):
    monkeypatch.setattr(server, "_last_used", time.monotonic() - 999)
    speak(server, text="Hello.")
    assert time.monotonic() - server._last_used < 5


def fake_native(server, monkeypatch):
    def fake_omni_take(text, language, instruct, seed, speed, out, ref_audio, ref_text):
        sf.write(out, voice_like(f0=200), SR)
        return {"spoken_text": text, "sentences": [], "duration_s": 2.0}
    monkeypatch.setattr(server, "omni_take", fake_omni_take)
    monkeypatch.setattr(server, "omni", lambda: FakeOmni())


def test_romanian_kinds_reach_their_own_stored_takes(server, monkeypatch):
    fake_native(server, monkeypatch)
    native = speak(server, text="Bună seara.", language="ro", voice="young-woman")             # auto: native
    clone = speak(server, text="Bună seara.", language="ro", voice="young-woman", ro_voice="clone")
    v = server.VOICES
    assert (v / "young-woman.ro.wav").is_file() and (v / "young-woman.wav").is_file()
    assert native["like"] == json.loads((v / "young-woman.ro.json").read_text(encoding="utf-8"))["id"]
    assert clone["like"] == json.loads((v / "young-woman.json").read_text(encoding="utf-8"))["id"]
    assert server._fakes[0].calls[-1]["text"] == server.VOICE_TEXT  # the clone came from an English take


def test_a_native_take_stored_before_kinds_is_moved_not_lost(server, monkeypatch):
    fake_native(server, monkeypatch)
    old = make_take(server, "20260101-120000-0001", text="Am așteptat.", language="ro")
    server.VOICES.mkdir()
    server._store("young-woman", old)                     # where the old layout kept her native voice
    assert server.default_voice_info("young-woman", "native")["id"] == old
    assert server.default_voice_info("young-woman", "clone") == {}
    res = speak(server, text="Bună.", language="ro", voice="young-woman")
    assert res["like"] == old and res["ro_voice"] == "native"  # the picked take keeps speaking


def test_auto_follows_the_saved_romanian_choice(server):
    assert server.resolve_kind("young-woman") == "native"
    server.save_settings(server.Settings(ro_voice="clone"))
    assert server.resolve_kind("young-woman") == "clone"
    c = client(server)
    assert c.get("/api/default-voice?voice=young-woman").json()["kind"] == "clone"


def test_a_romanian_take_stored_as_a_clone_stays_a_clone(server):
    """Review finding: the migration must not move takes stored after it."""
    c = client(server)
    ro = make_take(server, "20260101-120000-0002", text="Bună.", language="ro")
    assert c.post("/api/default-voice", json={"id": ro, "voice": "child", "kind": "clone"}).json()["id"] == ro
    assert c.get("/api/default-voice?voice=child&kind=clone").json()["id"] == ro
    yw = make_take(server, "20260101-120000-0003", text="Bună.", language="ro")
    c.post("/api/default-voice", json={"id": yw, "voice": "young-woman", "kind": "clone"})
    assert c.get("/api/default-voice?voice=young-woman&kind=clone").json()["id"] == yw  # not moved later either


def test_the_api_resolves_auto_to_the_kind_romanian_lines_use(server):
    c = client(server)
    first = make_take(server, "20260101-120000-0004", text="Bună.", language="ro")
    second = make_take(server, "20260101-120000-0005", text="Bună.", language="ro")
    r = c.post("/api/default-voice", json={"id": first, "voice": "young-woman"}).json()   # no kind: auto
    assert r["kind"] == "native" and (server.VOICES / "young-woman.ro.wav").is_file()
    assert not (server.VOICES / "young-woman.wav").exists()
    c.post("/api/default-voice", json={"id": second, "voice": "young-woman"})
    back = c.post("/api/default-voice/restore-previous", json={"voice": "young-woman"}).json()
    assert back["id"] == first and back["kind"] == "native"
    r = c.post("/api/default-voice", json={"id": first, "voice": "child"}).json()          # auto for child: clone
    assert r["kind"] == "clone" and (server.VOICES / "child.wav").is_file()
