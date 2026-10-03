r"""Smoke tests for voicefx: every preset and op runs and produces sane audio.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import numpy as np
import pytest

import voicefx
from voicefx import fx


def fake_voice(sr=24000, seconds=1.5):
    """A buzzy 110 Hz tone with a moving formant: close enough to exercise the chain."""
    t = np.arange(int(sr * seconds)) / sr
    src = np.sign(np.sin(2 * np.pi * 110 * t)) * 0.3
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    return (src * am).astype(np.float32), sr


@pytest.mark.parametrize("name", sorted(voicefx.list_presets()))
def test_preset_runs(name):
    x, sr = fake_voice()
    y = voicefx.apply(x, sr, voicefx.load_preset(name))
    assert y.dtype == np.float32 and y.ndim == 1
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 0.892  # peaks held under -1 dBFS
    assert abs(fx.loudness(y) - fx.TARGET_LUFS) < 0.5  # every preset at the same loudness
    assert len(y) >= len(x) * fx.SR / sr * 0.85  # the tail is trimmed; nothing collapses


@pytest.mark.parametrize("op", sorted(fx.OPS))
def test_every_op_runs(op):
    x, sr = fake_voice()
    y = fx.chain(fx.to_sr(x, sr), [{"op": op}])
    assert np.isfinite(y).all() and len(y) > 0


def test_unknown_op_is_an_error():
    with pytest.raises(ValueError):
        fx.chain(np.zeros(100, dtype=np.float32), [{"op": "nope"}])


def test_preset_names_only():
    with pytest.raises(FileNotFoundError):
        voicefx.load_preset("../../models.json")
    with pytest.raises(FileNotFoundError):
        voicefx.load_preset("no-such-preset")


@pytest.mark.parametrize("name", ["x/../reaper", "../presets/reaper", "a/../../presets/reaper", "reaper.json", ""])
def test_preset_names_must_be_bare(name):
    with pytest.raises(FileNotFoundError):
        voicefx.load_preset(name)


def test_edge_inputs():
    p = voicefx.load_preset("reaper-drone")
    assert len(voicefx.apply(np.zeros(0, dtype=np.float32), 24000, p)) == 0
    assert not np.any(voicefx.apply(np.zeros(24000, dtype=np.float32), 24000, p))       # silence stays silent
    assert not np.any(voicefx.apply(np.full(2400, 1e-7, dtype=np.float32), 24000, p))  # no noise blown up
    x, sr = fake_voice()
    x[100:200] = np.nan
    assert np.isfinite(voicefx.apply(x, sr, p)).all()


def test_stretch_inside_a_bus_is_not_cut_short():
    x, sr = fake_voice(seconds=1.0)
    p = {"buses": [{"name": "slow", "chain": [{"op": "stretch", "rate": 0.5}]}]}
    assert len(voicefx.apply(x, sr, p)) > 1.8 * fx.SR


def test_pitch_and_tempo():
    x, sr = fake_voice(seconds=1.0)
    p = voicefx.load_preset("dry")
    base = voicefx.apply(x, sr, p)
    slow = voicefx.apply(x, sr, p, tempo=0.5)
    assert len(slow) > 1.7 * len(base)
    low = voicefx.apply(x, sr, p, pitch=-12)
    spec = lambda y: np.abs(np.fft.rfft(y[: fx.SR]))  # noqa: E731
    f = np.fft.rfftfreq(fx.SR, 1 / fx.SR)
    centroid = lambda y: float(np.sum(f * spec(y)) / np.sum(spec(y)))  # noqa: E731
    assert centroid(low) < 0.75 * centroid(base)  # an octave down moves the energy down


@pytest.mark.parametrize("kind", ["room", "hall", "cathedral"])
def test_reverb_adds_a_tail(kind):
    x, sr = fake_voice(seconds=1.0)
    p = voicefx.load_preset("dry")
    dry = voicefx.apply(x, sr, p)
    wet = voicefx.apply(x, sr, p, reverb=kind)
    assert np.isfinite(wet).all() and len(wet) > len(dry)


def test_reverb_tails_grow_with_size():
    x, sr = fake_voice(seconds=1.0)
    p = voicefx.load_preset("dry")
    lens = [len(voicefx.apply(x, sr, p, reverb=k)) for k in ("room", "hall", "cathedral")]
    assert lens == sorted(lens)


def test_unknown_reverb_is_an_error():
    x, sr = fake_voice()
    with pytest.raises(ValueError):
        voicefx.apply(x, sr, voicefx.load_preset("dry"), reverb="stadium")


def test_loudness_matches_the_bs1770_reference():
    # The standard's calibration point: a 997 Hz sine at full scale reads -3.01 LUFS.
    t = np.arange(5 * fx.SR) / fx.SR
    sine = 0.1 * np.sin(2 * np.pi * 997 * t)
    assert fx.loudness(sine) == pytest.approx(-23.01, abs=0.1)


def test_presets_end_at_the_same_loudness():
    x, sr = fake_voice(seconds=3.0)
    levels = [fx.loudness(voicefx.apply(x, sr, voicefx.load_preset(n))) for n in voicefx.list_presets()]
    assert max(levels) - min(levels) < 0.5
