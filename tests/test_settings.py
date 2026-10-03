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
    monkeypatch.setattr(server, "OLD_VOICE_FILES", (tmp_path / "output" / "default_voice.json",
                                                    tmp_path / "output" / "house_voice.json"))
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


def test_old_pointer_in_output_is_migrated(app):
    tid = make_take(app, "20260101-120000-cccc")
    (app.OUTPUT / "house_voice.json").write_text(json.dumps({"id": tid, "f0_hz": 65.0}), encoding="utf-8")
    assert app.default_voice_info()["id"] == tid
    assert not (app.OUTPUT / "house_voice.json").exists() and (app.VOICES / "default.wav").is_file()


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
