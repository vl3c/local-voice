r"""Saved settings and the default voice's storage (no models are loaded).

    .venv\Scripts\python.exe -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import app as server  # noqa: PLC0415  (imports FastAPI and voicefx, loads no models)
    monkeypatch.setattr(server, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(server, "VOICES", tmp_path / "voices")
    monkeypatch.setattr(server, "SETTINGS_FILE", tmp_path / "config.local.json")
    (tmp_path / "output").mkdir()
    return server


def make_take(app, tid: str, f0: float = 65.0, text: str = "We are the dark.") -> str:
    sr = 24000
    t = np.arange(sr * 2) / sr
    voice = 0.3 * np.sign(np.sin(2 * np.pi * f0 * t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    sf.write(app.take_paths(tid)["dry"], voice.astype(np.float32), sr)
    app.take_paths(tid)["meta"].write_text(json.dumps({"id": tid, "text": text, "language": "en"}), encoding="utf-8")
    return tid


# ------------------------------------------------------------------ settings

def test_no_saved_settings_means_repo_defaults(app):
    assert app.local_settings() == {}
    assert app.effective_settings() == app.REPO_DEFAULTS


def test_saving_keeps_only_what_differs_from_the_repo(app):
    app.save_settings(app.Settings(preset="harbinger", seed=app.DEFAULT_SEED, pitch=-3, reverb="room"))
    saved = json.loads(app.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert saved == {"preset": "harbinger", "pitch": -3}  # seed and reverb equal the repo's: not stored
    eff = app.effective_settings()
    assert eff["preset"] == "harbinger" and eff["pitch"] == -3 and eff["seed"] == app.DEFAULT_SEED


def test_random_seed_is_a_real_choice(app):
    app.save_settings(app.Settings(seed=None))
    assert app.effective_settings()["seed"] is None


def test_unknown_preset_is_refused(app):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        app.save_settings(app.Settings(preset="no-such-preset"))
    assert not app.SETTINGS_FILE.exists()


def test_a_broken_settings_file_falls_back_to_the_repo(app):
    app.SETTINGS_FILE.write_text('{"pitch": 99}', encoding="utf-8")  # out of range
    assert app.effective_settings() == app.REPO_DEFAULTS


def test_reset_removes_the_saved_settings(app):
    app.save_settings(app.Settings(pitch=-2))
    app.reset_settings()
    assert not app.SETTINGS_FILE.exists() and app.effective_settings() == app.REPO_DEFAULTS


def test_requests_take_saved_settings_only_for_what_they_leave_out(app):
    app.save_settings(app.Settings(preset="harbinger", pitch=-4))
    req = app.SpeakReq(text="x", pitch=2)
    opt = app._resolve(req, ("preset", "pitch", "reverb"))
    assert opt == {"preset": "harbinger", "pitch": 2, "reverb": app.DEFAULT_REVERB}


# ------------------------------------------------------------------ default voice

def test_default_voice_survives_a_cleared_output(app):
    tid = make_take(app, "20260101-120000-aaaa")
    app.set_default_voice(tid)
    for p in app.OUTPUT.iterdir():
        p.unlink()
    info = app.default_voice_info()
    assert info["id"] == tid and info["text"] == "We are the dark."
    wav, text, source = app.default_voice_ref()
    assert wav.is_file() and text == "We are the dark." and source == tid


def test_previous_voice_is_remembered_and_restorable(app):
    a = make_take(app, "20260101-120000-aaaa", f0=65)
    b = make_take(app, "20260101-120000-bbbb", f0=80, text="We are the end.")
    app.set_default_voice(a)
    info = app.set_default_voice(b)
    assert info["id"] == b and info["previous"]["id"] == a
    for p in app.OUTPUT.iterdir():  # undo must not need the takes
        p.unlink()
    back = app.restore_previous_voice()
    assert back["id"] == a and back["previous"]["id"] == b and back["text"] == "We are the dark."


def test_setting_the_same_voice_twice_keeps_the_real_previous(app):
    a = make_take(app, "20260101-120000-aaaa")
    b = make_take(app, "20260101-120000-bbbb")
    app.set_default_voice(a)
    app.set_default_voice(b)
    assert app.set_default_voice(b)["previous"]["id"] == a


# ------------------------------------------------------------------ takes

def test_trim_edges_drops_a_leading_blip(app):
    sr = 24000
    blip = np.r_[np.zeros(int(0.1 * sr)), 0.3 * np.random.default_rng(0).standard_normal(int(0.05 * sr)),
                 np.zeros(int(0.4 * sr))]
    t = np.arange(int(1.0 * sr)) / sr
    speech = 0.3 * np.sin(2 * np.pi * 120 * t)
    clip = np.r_[blip, speech, np.zeros(int(0.3 * sr))].astype(np.float32)
    out = app.trim_edges(clip, sr)
    assert len(out) < len(clip) - int(0.5 * sr)          # the blip, its silence and the tail are gone
    assert np.max(np.abs(out[: int(0.02 * sr)])) < 0.01   # it starts at the speech, not the blip
    assert len(out) >= int(1.0 * sr)                      # all the speech is still there


def test_deleting_a_take_removes_only_its_files(app):
    a = make_take(app, "20260101-120000-aaaa")
    b = make_take(app, "20260101-120000-bbbb")
    (app.OUTPUT / f"{a}_fx-reaper-drone.mp3").write_bytes(b"x")
    app.set_default_voice(a)
    assert app.delete_take(a)["deleted"] == 3
    assert not list(app.OUTPUT.glob(f"{a}*")) and app.take_paths(b)["dry"].is_file()
    assert app.default_voice_info()["id"] == a            # the default voice is a copy in voices/


def test_deleting_needs_a_valid_take_id(app):
    from fastapi import HTTPException
    for bad in ("../x", "*", "20260101-120000-zzzz"):
        with pytest.raises(HTTPException):
            app.delete_take(bad)


# ------------------------------------------------------------------ review fixes

def test_cleared_number_fields_are_not_saved_as_empty(app):
    app.save_settings(app.Settings(cfg_scale=None, tempo=None, pitch=-2, seed=None))
    saved = app.local_settings()
    assert saved == {"pitch": -2, "seed": None}           # nulls dropped, except seed = random
    eff = app.effective_settings()
    assert eff["cfg_scale"] == app.DEFAULT_CFG and eff["tempo"] == 1.0


def test_settings_file_with_nulls_falls_back_per_key(app):
    app.SETTINGS_FILE.write_text('{"tempo": null, "pitch": -3}', encoding="utf-8")
    eff = app.effective_settings()
    assert eff["tempo"] == 1.0 and eff["pitch"] == -3


def test_render_deletion_never_touches_the_take(app):
    from fastapi import HTTPException
    tid = make_take(app, "20260101-120000-aaaa")
    render = app.OUTPUT / f"{tid}_fx-reaper-drone_room.mp3"
    render.write_bytes(b"x")
    assert app.delete_render(render.name)["deleted"] == 1 and not render.exists()
    for bad in (f"{tid}_dry.wav", f"{tid}.json", "../x_fx-a.wav", f"{tid}_fx-a.txt"):
        with pytest.raises(HTTPException):
            app.delete_render(bad)
    assert app.take_paths(tid)["dry"].is_file()


def test_cross_site_writes_are_refused(app):
    from fastapi.testclient import TestClient
    c = TestClient(app.app, base_url="http://127.0.0.1:8190")
    evil = c.post("/api/default-voice/restore-previous", headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    sneaky = c.post("/api/default-voice/restore-previous", headers={"Sec-Fetch-Site": "cross-site"})
    assert sneaky.status_code == 403
    page = c.post("/api/default-voice/restore-previous", headers={"Origin": "http://127.0.0.1:8190"})
    assert page.status_code == 404                         # allowed through: there's simply no previous voice
    cli = c.post("/api/default-voice/restore-previous")
    assert cli.status_code == 404
    assert c.get("/api/settings").headers["X-Frame-Options"] == "DENY"


def test_take_ids_must_match_exactly(app):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        app.load_take("20260101-120000-aaaa\n")


# ------------------------------------------------------------------ voices

class FakeBreeze:
    """Records what speak() asks the model for; returns a short buzz."""
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def synth(self, text, instruction, seed, cfg_scale, ref_audio=None, ref_text=None):
        self.calls.append({"instruction": instruction, "seed": seed, "ref": ref_audio})
        t = np.arange(self.sample_rate) / self.sample_rate
        return (0.3 * np.sign(np.sin(2 * np.pi * 110 * t))).astype(np.float32), cfg_scale


@pytest.fixture()
def fake(app, monkeypatch):
    engine = FakeBreeze()
    monkeypatch.setattr(app, "tts", lambda: engine)
    return engine


def say(app, **body):
    return app.speak(app.SpeakReq(**{"text": "We are the dark.", "check": False, **body}))


def test_no_voice_means_the_default_voice_with_its_seed(app, fake):
    r = say(app)
    v = app.voice_catalog()[app.DEFAULT_VOICE]
    assert fake.calls[-1]["instruction"] == v["instruction"] and fake.calls[-1]["seed"] == v["seed"]
    assert r["voice"] == app.DEFAULT_VOICE


def test_a_voice_brings_its_own_description_and_seed(app, fake):
    v = app.voice_catalog()["young-woman"]
    say(app, voice="young-woman")
    assert fake.calls[-1] == {"instruction": v["instruction"], "seed": v["seed"], "ref": None}
    say(app, voice="young-woman", seed=5)
    assert fake.calls[-1]["seed"] == 5  # an explicit seed still wins


def test_a_description_overrides_the_voice(app, fake):
    r = say(app, voice="child", instruction="A tired robot.")
    assert fake.calls[-1]["instruction"] == "A tired robot." and r["voice"] is None


def test_the_saved_voice_applies_when_the_request_names_none(app, fake):
    app.save_settings(app.Settings(voice="child"))
    say(app)
    assert fake.calls[-1]["instruction"] == app.voice_catalog()["child"]["instruction"]
    assert fake.calls[-1]["seed"] == app.voice_catalog()["child"]["seed"]


def test_unknown_voices_are_refused(app, fake):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        say(app, voice="no-such-voice")
    with pytest.raises(HTTPException):
        app.save_settings(app.Settings(voice="no-such-voice"))


def test_each_voice_keeps_its_own_romanian_clone(app):
    make_take(app, "20260101-120000-aaaa", f0=65)
    make_take(app, "20260101-120000-bbbb", f0=280)
    app.set_default_voice("20260101-120000-aaaa")
    app.set_default_voice("20260101-120000-bbbb", "child")
    assert app.default_voice_info()["id"] == "20260101-120000-aaaa"
    assert app.default_voice_info("child")["id"] == "20260101-120000-bbbb"
    assert app.default_voice_info("young-woman") == {}


def test_presets_no_longer_carry_a_voice():
    import voicefx
    assert all("instruction" not in p for p in voicefx.list_presets().values())


def test_a_voice_with_romanian_attributes_gets_a_native_romanian_take(app, monkeypatch):
    made = []

    def fake_omni_take(text, language, instruct, seed, speed, out, ref_audio, ref_text):
        made.append({"language": language, "instruct": instruct, "seed": seed, "ref": ref_audio})
        sr = 24000
        t = np.arange(sr) / sr
        sf.write(out, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
        return {"spoken_text": text, "sentences": [], "duration_s": 1.0}

    monkeypatch.setattr(app, "omni_take", fake_omni_take)
    monkeypatch.setattr(app, "tts", lambda: pytest.fail("a native Romanian voice needs no English take"))
    spec = app.voice_catalog()["young-woman"]["romanian"]
    wav, text, tid = app.default_voice_ref("young-woman", "native")
    assert made == [{"language": "ro", "instruct": spec["attributes"], "seed": spec["seed"], "ref": None}]
    assert wav.is_file() and text == app.VOICE_TEXT_RO
    assert app.default_voice_info("young-woman", "native")["id"] == tid
    assert app.default_voice_info("young-woman", "clone") == {}  # kept apart from the English clone
    assert app.default_voice_ref("young-woman", "native")[2] == tid  # made once, then reused


def test_a_saved_english_description_never_reaches_romanian(app, monkeypatch):
    """Review finding: the saved description is English prose; OmniVoice would read it as attributes."""
    ref = make_take(app, "20260101-120000-cccc")
    calls = []

    def fake_omni_take(text, language, instruct, seed, speed, out, ref_audio, ref_text):
        calls.append({"instruct": instruct, "ref": ref_audio})
        sr = 24000
        sf.write(out, (0.3 * np.sin(2 * np.pi * 120 * np.arange(sr) / sr)).astype(np.float32), sr)
        return {"spoken_text": text, "sentences": [], "duration_s": 1.0}

    monkeypatch.setattr(app, "omni_take", fake_omni_take)
    monkeypatch.setattr(app, "default_voice_ref",
                        lambda voice, kind="clone": (app.take_paths(ref)["dry"], "We are the dark.", ref))
    app.save_settings(app.Settings(instruction="A gravelly old pirate, loud and slurring."))
    r = app.speak(app.SpeakReq(text="Bună ziua.", language="ro", check=False))
    assert calls[-1]["instruct"] is None and calls[-1]["ref"] is not None  # cloned, not designed
    assert r["default_voice"] is True


def test_romanian_voice_choice(app):
    from fastapi import HTTPException
    assert app.romanian_kind("ancient-male") == "clone"           # auto: the catalog's choice
    assert app.romanian_kind("young-woman") == "native"           # her clone had an English accent
    assert app.romanian_kind("young-woman", "clone") == "clone"   # forced either way
    assert app.romanian_kind("vampire-female", "native") == "native"
    for v in app.voice_catalog():
        assert app.romanian_kind(v, "native") == "native"         # every voice has a native one
    app.VOICE_CATALOG_BACKUP = app.voice_catalog
    try:
        app.voice_catalog = lambda: {"x": {"instruction": "A voice.", "seed": 1}}
        assert app.romanian_kind("x") == "clone"
        with pytest.raises(HTTPException):
            app.romanian_kind("x", "native")                       # no attributes: no native voice
    finally:
        app.voice_catalog = app.VOICE_CATALOG_BACKUP


def test_romanian_requests_use_the_chosen_kind(app, monkeypatch):
    ref = make_take(app, "20260101-120000-dddd")
    asked = []

    def fake_ref(voice, kind="clone"):
        asked.append((voice, kind))
        return app.take_paths(ref)["dry"], "We are the dark.", ref

    def fake_omni_take(text, language, instruct, seed, speed, out, ref_audio, ref_text):
        sr = 24000
        sf.write(out, (0.3 * np.sin(2 * np.pi * 120 * np.arange(sr) / sr)).astype(np.float32), sr)
        return {"spoken_text": text, "sentences": [], "duration_s": 1.0}

    monkeypatch.setattr(app, "default_voice_ref", fake_ref)
    monkeypatch.setattr(app, "omni_take", fake_omni_take)
    r = app.speak(app.SpeakReq(text="Bună.", language="ro", voice="young-woman", check=False))
    assert asked[-1] == ("young-woman", "native") and r["ro_voice"] == "native"
    r = app.speak(app.SpeakReq(text="Bună.", language="ro", voice="young-woman", ro_voice="clone", check=False))
    assert asked[-1] == ("young-woman", "clone") and r["ro_voice"] == "clone"
    app.save_settings(app.Settings(ro_voice="native"))
    app.speak(app.SpeakReq(text="Bună.", language="ro", voice="vampire-male", check=False))
    assert asked[-1] == ("vampire-male", "native")                # the saved choice applies
