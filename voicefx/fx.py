"""Sound-design effects that turn a TTS voice into something that is not human.

A preset (voicefx/presets/*.json) is a small signal graph:

    pre    one chain run on the dry voice (slow it down, clean it up, compress);
           its `stretch` ops always run first, as one stretch with the take's tempo,
           so a clean copy of the voice stays aligned with the result (apply_layers)
    buses  parallel chains, each fed the output of `pre`, mixed at `gain_db`; a bus
           with "stereo_echo" isn't mixed in but bounces left-right behind the
           voice (see stereo_echo), which makes the take stereo
    post   one chain run on the mix (distortion, EQ, space, limiter)

Each chain is a list of ops, e.g. {"op": "pitch", "semitones": -12}. OPS below
lists them all. Everything runs at 48 kHz mono float32 on the CPU; a few seconds
of speech takes well under a second.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pedalboard as pb
from scipy import signal

SR = 48000
PRESET_DIR = Path(__file__).resolve().parent / "presets"


# --------------------------------------------------------------------------- helpers

def db(x: float) -> float:
    return float(10 ** (x / 20))


def to_sr(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono audio to SR."""
    if sr == SR:
        return audio.astype(np.float32)
    g = np.gcd(sr, SR)
    return signal.resample_poly(audio, SR // g, sr // g).astype(np.float32)


def run(board_or_plugins, x: np.ndarray) -> np.ndarray:
    """Run plugins over mono audio, or stereo as (2, samples)."""
    board = board_or_plugins if isinstance(board_or_plugins, pb.Pedalboard) else pb.Pedalboard(board_or_plugins)
    return board(x, SR, reset=True) if x.ndim == 2 else board(x[None, :], SR, reset=True)[0]


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


# --------------------------------------------------------------------------- ops

def op_gain(x, db_=0.0, **_):
    return x * db(db_)


def op_pitch(x, semitones=0.0, preserve_formants=False, **_):
    """Shift pitch, keeping duration.

    preserve_formants=False (the default) moves the formants with the pitch, so
    a downward shift sounds like a physically larger throat, which is the point.
    """
    if not semitones:
        return x
    y = pb.time_stretch(x[None, :], SR, stretch_factor=1.0,
                        pitch_shift_in_semitones=float(semitones),
                        preserve_formants=bool(preserve_formants), high_quality=True)[0]
    return fit(y, len(x))


def op_stretch(x, rate=1.0, **_):
    """Change tempo without changing pitch. rate < 1 is slower."""
    if rate == 1.0:
        return x
    return pb.time_stretch(x[None, :], SR, stretch_factor=float(rate),
                           preserve_formants=True, high_quality=True)[0]


def op_formant(x, ratio=1.0, pitch_floor=50.0, **_):
    """Scale the formants alone (Praat 'Change gender'). ratio < 1 = bigger body."""
    if ratio == 1.0:
        return x
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    out = call(snd, "Change gender", float(pitch_floor), 600.0, float(ratio), 0.0, 1.0, 1.0)
    return fit(out.values[0].astype(np.float32), len(x))


def op_highpass(x, hz=60.0, **_):
    return run([pb.HighpassFilter(cutoff_frequency_hz=float(hz))], x)


def op_lowpass(x, hz=8000.0, **_):
    return run([pb.LowpassFilter(cutoff_frequency_hz=float(hz))], x)


def op_eq(x, bands=(), **_):
    """bands: [{"type": "lowshelf|highshelf|peak|lowpass|highpass", "hz", "gain_db", "q"}]."""
    plugins = []
    for b in bands:
        t, hz = b["type"], float(b["hz"])
        g, q = float(b.get("gain_db", 0.0)), float(b.get("q", 0.707))
        if t == "lowshelf":
            plugins.append(pb.LowShelfFilter(cutoff_frequency_hz=hz, gain_db=g, q=q))
        elif t == "highshelf":
            plugins.append(pb.HighShelfFilter(cutoff_frequency_hz=hz, gain_db=g, q=q))
        elif t == "peak":
            plugins.append(pb.PeakFilter(cutoff_frequency_hz=hz, gain_db=g, q=q))
        elif t == "lowpass":
            plugins.append(pb.LowpassFilter(cutoff_frequency_hz=hz))
        elif t == "highpass":
            plugins.append(pb.HighpassFilter(cutoff_frequency_hz=hz))
        else:
            raise ValueError(f"unknown eq band type {t!r}")
    return run(plugins, x) if plugins else x


def op_compress(x, threshold_db=-30.0, ratio=4.0, attack_ms=5.0, release_ms=120.0, makeup_db=0.0, **_):
    y = run([pb.Compressor(threshold_db=float(threshold_db), ratio=float(ratio),
                           attack_ms=float(attack_ms), release_ms=float(release_ms))], x)
    return y * db(makeup_db)


DRIVE_REF_DB = -20.0  # drive_db is applied to a signal first brought to this RMS


def op_drive(x, drive_db=12.0, mix=1.0, **_):
    """Distortion, blended in parallel with the clean signal.

    The input is normalised to DRIVE_REF_DB RMS first, so a given drive_db
    distorts the same amount whatever level the chain happens to be at.
    """
    level = _rms(x)
    norm = db(DRIVE_REF_DB) / max(level, 1e-9)
    wet = run([pb.Distortion(drive_db=float(drive_db))], x * norm)
    # Bring the wet back to the dry's RMS before mixing.
    wet *= level / max(_rms(wet), 1e-9)
    return (1 - mix) * x + mix * wet


def op_ringmod(x, hz=80.0, mix=0.3, **_):
    """Multiply by a sine: inharmonic, metallic sidebands."""
    t = np.arange(len(x), dtype=np.float32) / SR
    wet = x * np.sin(2 * np.pi * float(hz) * t).astype(np.float32)
    return (1 - mix) * x + mix * wet * np.sqrt(2)


def op_comb(x, delay_ms=4.0, feedback=0.6, mix=0.5, **_):
    """Feedback comb filter: a resonant metal tube."""
    d = max(1, int(SR * float(delay_ms) / 1000))
    a = np.zeros(d + 1, dtype=np.float64)
    a[0], a[d] = 1.0, -float(feedback)
    wet = signal.lfilter([1.0 - float(feedback)], a, x).astype(np.float32)
    return (1 - mix) * x + mix * wet


def op_chorus(x, rate_hz=1.0, depth=0.25, delay_ms=7.0, feedback=0.0, mix=0.5, **_):
    return run([pb.Chorus(rate_hz=float(rate_hz), depth=float(depth), centre_delay_ms=float(delay_ms),
                          feedback=float(feedback), mix=float(mix))], x)


def _lowpass_sos(hz: float, order: int = 2):
    return signal.butter(order, float(hz) / (SR / 2), output="sos")


def _pitch_track(x: np.ndarray, floor: float = 40.0, ceiling: float = 300.0) -> np.ndarray | None:
    """Per-sample f0 of the voice, with unvoiced gaps bridged; None if nothing is voiced."""
    import parselmouth
    p = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR).to_pitch(
        time_step=0.01, pitch_floor=float(floor), pitch_ceiling=float(ceiling))
    f, t = p.selected_array["frequency"], p.xs()
    voiced = f > 0
    if voiced.sum() < 3:
        return None
    f = np.interp(t, t[voiced], f[voiced])
    return np.interp(np.arange(len(x)) / SR, t, f)


def _carrier_f0(x: np.ndarray, f0, follow: float, steps: float, f0_ratio: float = 1.0) -> np.ndarray:
    """The machine's pitch over time.

    follow=0 holds f0 dead steady (a drone that speaks); follow=1 copies the
    voice's intonation, transposed so its median lands on f0. steps > 0 snaps
    the result to a grid of that many semitones: the stepped, quantised
    melody of a machine imitating speech. f0="voice" puts the machine at the
    speaker's own median pitch (times f0_ratio): a machine that sings along
    with any voice, high or low, rather than a fixed drone.
    """
    n = len(x)
    track = _pitch_track(x, ceiling=600.0) if (follow > 0 or f0 == "voice") else None
    if f0 == "voice":
        f0 = (float(np.median(track)) if track is not None else 120.0) * float(f0_ratio)
    else:
        f0 = float(f0) * float(f0_ratio)
    if track is None or not follow:
        return np.full(n, float(f0))
    semis = 12 * np.log2(track / np.median(track)) * float(follow)
    if steps > 0:
        semis = np.round(semis / steps) * steps
    semis = signal.sosfiltfilt(_lowpass_sos(25.0), semis)  # slew, so steps glide instead of click
    return float(f0) * 2 ** (semis / 12)


def brass(freq: np.ndarray, chord=(1.0,), detune=(-6.0, 0.0, 7.0), metal: float = 0.0,
          sub: float = 0.0, seed: int = 0) -> np.ndarray:
    """Detuned sawtooth stack: a brass section of machines, at a per-sample pitch.

    metal adds inharmonic 'scraped steel' partials; sub adds a square
    an octave below.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(len(freq))
    for r in chord:
        for c in detune:
            ph = np.cumsum(freq * float(r) * 2 ** (c / 1200)) / SR + rng.uniform()
            out += 2 * (ph % 1.0) - 1
    out /= len(chord) * len(detune)
    if sub > 0:
        ph = np.cumsum(freq / 2) / SR
        out += float(sub) * np.sign(np.sin(2 * np.pi * ph)) * 0.5
    if metal > 0:
        scrape = np.zeros(len(freq))
        for ratio in (6.3, 9.1, 13.7, 17.2, 23.9, 31.4):
            scrape += np.sin(2 * np.pi * np.cumsum(freq * ratio) / SR + rng.uniform(0, 6.28)) / np.sqrt(ratio)
        out += float(metal) * scrape * (_rms(out) / max(_rms(scrape), 1e-9))
    return out


def op_vocoder(x, f0=43.65, chord=(1.0,), follow=0.0, steps=0.0, bands=36, fmin=50.0, fmax=8000.0,
               noise=0.1, fricatives=1.0, metal=0.0, sub=0.0, smooth_ms=10.0, phase="zero",
               tone=1.0, f0_ratio=1.0, **_):
    """Channel vocoder: the voice's band envelopes imposed on a synthetic carrier.

    The carrier is a brass machine tone (see `brass`) at f0 - steady, or
    following the voice by `follow`, optionally quantised to `steps` semitones -
    so the machine itself does the speaking. Where the voice is hissing (s, sh,
    f, t) the carrier crossfades to noise, scaled by `fricatives`, which keeps
    consonants intelligible; `noise` is a constant breath under everything.
    tone scales the machine tone itself: tone=0 with noise=1 leaves only breath,
    a whisper shaped by the words. f0 may be "voice" (see _carrier_f0).

    phase="zero" (the default) filters forwards and backwards, so the machine
    speaks exactly in time with the voice. phase="causal" is the original
    one-pass filtering: the same bands, but each arrives late by its filters'
    delay (~80 ms under 200 Hz, ~25 ms at 300 Hz-1 kHz, ~10 ms at 1-4 kHz), so
    the result can't be blended with the voice that drove it.
    """
    zero = phase == "zero"
    if phase not in ("zero", "causal"):
        raise ValueError(f"vocoder phase must be 'zero' or 'causal', not {phase!r}")
    # Forward-backward doubles each filter's slope, so the zero-phase path uses half the order:
    # the same steepness, no delay.
    filt = signal.sosfiltfilt if zero else signal.sosfilt
    order = 2 if zero else 4
    n = len(x)
    tone_level = float(tone)
    tone = brass(_carrier_f0(x, f0, follow, steps, f0_ratio), chord=chord, metal=metal, sub=sub)
    tone *= 0.577 / max(_rms(tone), 1e-9)  # a plain saw's RMS, so noise levels mean the same thing
    hiss = np.random.default_rng(7).standard_normal(n) * 0.577

    # How fricative the voice is right now: high-band share of its energy.
    env_lp = _lowpass_sos(1000.0 / (np.pi * 15.0), order // 2)
    e_hi = filt(env_lp, np.abs(filt(signal.butter(order, 3500 / (SR / 2), "high", output="sos"), x)))
    e_all = filt(env_lp, np.abs(x))
    u = np.clip((e_hi / (e_all + 1e-6) - 0.15) / 0.35, 0, 1) * float(fricatives)
    carrier = tone_level * tone * (1 - np.clip(u, 0, 1)) + hiss * (u + float(noise))

    edges = np.geomspace(fmin, fmax, int(bands) + 1)
    lp = _lowpass_sos(1000.0 / (np.pi * smooth_ms), order // 2)  # envelope smoother
    out = np.zeros(n)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sos = signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
        env = filt(lp, np.abs(filt(sos, x)))
        out += filt(sos, carrier) * np.maximum(env, 0)
    out = out.astype(np.float32)
    return out * (_rms(x) / max(_rms(out), 1e-9))


def op_reverb(x, room_size=0.9, damping=0.5, wet=0.2, dry=0.8, width=1.0, **_):
    return run([pb.Reverb(room_size=float(room_size), damping=float(damping), wet_level=float(wet),
                          dry_level=float(dry), width=float(width))], x)


def op_delay(x, delay_ms=250.0, feedback=0.3, mix=0.2, **_):
    return run([pb.Delay(delay_seconds=float(delay_ms) / 1000, feedback=float(feedback), mix=float(mix))], x)


def op_crush(x, bits=8, downsample=1, mix=1.0, **_):
    """Lo-fi digital grit: fewer bits, and sample-and-hold to a lower rate (downsample=4
    is 12 kHz). Toy robots, broken radios, small machines."""
    d = max(1, int(downsample))
    wet = np.repeat(x[::d], d)[: len(x)] if d > 1 else x
    wet = run([pb.Bitcrush(bit_depth=float(bits))], wet.astype(np.float32))
    return (1 - mix) * x + mix * wet


def op_preverb(x, room_size=0.9, damping=0.5, mix=0.4, **_):
    """Reverse reverb: each sound swells in out of its own echo before it starts, the way
    a ghost breathes in. The words themselves stay where they are."""
    rev = run([pb.Reverb(room_size=float(room_size), damping=float(damping), wet_level=1.0, dry_level=0.0)],
              x[::-1].copy())[::-1]
    return (1 - mix) * x + mix * rev * (_rms(x) / max(_rms(rev), 1e-9))


def op_swarm(x, voices=({"semitones": 0.0},), **_):
    """Many voices at once: copies of the voice, each pitched by `semitones` (fractions are
    detune), shifted by `delay_ms` (negative = early, so the crowd centres on the original
    timing) and mixed at `gain_db`. The sum keeps the input's level."""
    out = np.zeros(len(x), dtype=np.float64)
    for v in voices:
        y = op_pitch(x, semitones=float(v.get("semitones", 0.0)))
        d = int(SR * float(v.get("delay_ms", 0.0)) / 1000)
        y = np.concatenate([np.zeros(d, np.float32), y])[: len(x)] if d > 0 else (
            np.concatenate([y[-d:], np.zeros(-d, np.float32)]) if d < 0 else y)
        out += y * db(float(v.get("gain_db", 0.0)))
    return (out * (_rms(x) / max(_rms(out), 1e-9))).astype(np.float32)


def op_arrive(x, sweep_s=0.45, level_db=-10.0, lo_hz=300.0, hi_hz=6000.0, gap_s=0.3, q=1.2, **_):
    """A swoosh into every phrase: filtered noise that swells and rises in pitch over
    `sweep_s` and lands exactly as the phrase starts, so the voice seems to fly in from
    somewhere. A phrase starts after `gap_s` of silence (and at the very start). The words
    keep their timing; only the space before them changes."""
    hop = int(0.01 * SR)
    n = len(x) // hop
    if n < 3:
        return x
    level = 10 * np.log10(np.mean(x[: n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1) + 1e-12)
    active = level > level.max() - 35
    gap = int(gap_s / 0.01)
    onsets, quiet = [], gap  # frames of silence before the current frame
    for i, a in enumerate(active):
        if a and quiet >= gap:
            onsets.append(i * hop)
        quiet = 0 if a else quiet + 1
    if not onsets:
        return x
    rng = np.random.default_rng(11)
    out = x.astype(np.float64).copy()
    mask = np.zeros(len(x), dtype=bool)
    mask[: n * hop] = active.repeat(hop)  # the last partial frame counts as quiet
    amp = _rms(x[mask] if mask.any() else x) * db(level_db)
    length = int(sweep_s * SR)
    for on in onsets:
        start = max(0, on - length)
        m = on - start + int(0.06 * SR)  # sweep up to the onset, then a 60 ms tail into the word
        if m <= 0:
            continue
        t = np.arange(m) / max(1, on - start)
        noise = rng.standard_normal(m)
        # A band-pass whose centre rises exponentially from lo_hz to hi_hz (state-variable filter).
        fc = lo_hz * (hi_hz / lo_hz) ** np.clip(t, 0, 1)
        f = 2 * np.sin(np.pi * np.minimum(fc, SR / 6) / SR)
        damp = 1.0 / q
        low = band = 0.0
        y = np.empty(m)
        for k in range(m):
            high = noise[k] - low - damp * band
            band += f[k] * high
            low += f[k] * band
            y[k] = band
        env = np.where(t <= 1, np.clip(t, 0, 1) ** 2.5, np.exp(-(t - 1) * (on - start) / (0.02 * SR)))
        y *= env
        y *= amp / max(_rms(y[env > 0.3]) if (env > 0.3).any() else _rms(y), 1e-9)
        end = min(len(out), start + m)
        out[start:end] += y[: end - start]
    return out.astype(np.float32)


def op_echo(x, delay_ms=500.0, feedback=0.35, mix=0.25, lowpass_hz=1800.0, repeats=4, dry=1.0, **_):
    """A muffled echo: the voice repeated every `delay_ms`, each repeat darker (low-passed
    at `lowpass_hz`) and quieter by `feedback`, mixed in at `mix`. A memory of what was said.
    dry=0 keeps only the repeats (an echo bus under a clean voice)."""
    d = int(SR * float(delay_ms) / 1000)
    if d <= 0 or len(x) <= d:
        return x
    sos = _lowpass_sos(float(lowpass_hz))
    out = x.astype(np.float64) * float(dry)
    tap = x.astype(np.float64)
    for k in range(1, int(repeats) + 1):
        tap = signal.sosfilt(sos, np.concatenate([np.zeros(d), tap])[: len(x)]) * (1.0 if k == 1 else float(feedback))
        out += float(mix) * tap
    return out.astype(np.float32)


def op_limit(x, threshold_db=-1.0, release_ms=200.0, **_):
    return run([pb.Limiter(threshold_db=float(threshold_db), release_ms=float(release_ms))], x)


OPS = {
    "gain": lambda x, **k: op_gain(x, db_=k.get("db", 0.0)),
    "pitch": op_pitch, "stretch": op_stretch, "formant": op_formant,
    "highpass": op_highpass, "lowpass": op_lowpass, "eq": op_eq,
    "compress": op_compress, "drive": op_drive, "ringmod": op_ringmod,
    "comb": op_comb, "chorus": op_chorus, "vocoder": op_vocoder,
    "reverb": op_reverb, "delay": op_delay, "limit": op_limit,
    "crush": op_crush, "preverb": op_preverb, "swarm": op_swarm,
    "arrive": op_arrive, "echo": op_echo,
}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def chain(x: np.ndarray, ops: list[dict]) -> np.ndarray:
    for spec in ops:
        spec = dict(spec)
        name = spec.pop("op")
        if spec.pop("off", False):
            continue
        if name not in OPS:
            raise ValueError(f"unknown op {name!r}; known: {', '.join(sorted(OPS))}")
        x = OPS[name](x, **spec).astype(np.float32)
    return x


# --------------------------------------------------------------------------- presets

def list_presets() -> dict[str, dict]:
    out = {}
    for p in sorted(PRESET_DIR.glob("*.json")):
        spec = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = spec
    return out


PRESET_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Optional space, applied last. The presets themselves are dry, so a mix can add its
# own room instead. (Freeverb settings; the cathedral also gets a short pre-delay so
# the first words stay clear of their own echo.)
REVERBS = {
    "none": None,
    "room": dict(room_size=0.62, damping=0.5, wet=0.16, dry=0.92, predelay_ms=0, tail_s=1.5),
    "hall": dict(room_size=0.86, damping=0.4, wet=0.26, dry=0.86, predelay_ms=20, tail_s=3.0),
    "cathedral": dict(room_size=0.94, damping=0.35, wet=0.24, dry=0.88, predelay_ms=40, tail_s=4.5),
}


def op_space(x: np.ndarray, kind: str) -> np.ndarray:
    r = REVERBS[kind]
    if r is None:
        return x
    x = np.pad(x, (0, int(r["tail_s"] * SR)))
    wet = x
    if r["predelay_ms"]:
        wet = np.concatenate([np.zeros(int(SR * r["predelay_ms"] / 1000), dtype=np.float32), x])[: len(x)]
    wet = run([pb.Reverb(room_size=r["room_size"], damping=r["damping"], wet_level=1.0, dry_level=0.0)], wet)
    return r["dry"] * x + r["wet"] * wet
SILENCE = 1e-4  # peak below this (-80 dBFS) is treated as silence, not normalised up


def load_preset(name: str) -> dict:
    """Load voicefx/presets/<name>.json. Bare names only: the name ends up in output file names."""
    p = PRESET_DIR / f"{name}.json"
    if not PRESET_NAME.match(name) or not p.is_file():
        raise FileNotFoundError(f"no preset {name!r}; have: {', '.join(list_presets())}")
    return json.loads(p.read_text(encoding="utf-8"))


def apply(audio: np.ndarray, sr: int, preset: dict, pitch: float = 0.0, tempo: float = 1.0,
          reverb: str = "none", target_lufs: float = -16.0, tail_s: float = 2.5,
          blend: float = 0.0, double_ms: float = 0.0, bass_db: float = 0.0) -> np.ndarray:
    """Run a preset over mono audio at any rate; returns 48 kHz float32: mono, or (2, samples)
    for a preset with a stereo echo (see apply_layers)."""
    return apply_layers(audio, sr, preset, pitch, tempo, reverb, target_lufs, tail_s, blend, double_ms, bass_db)[0]


def apply_layers(audio: np.ndarray, sr: int, preset: dict, pitch: float = 0.0, tempo: float = 1.0,
                 reverb: str = "none", target_lufs: float = -16.0, tail_s: float = 2.5,
                 blend: float = 0.0, double_ms: float = 0.0,
                 bass_db: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Run a preset over mono audio at any rate: (processed, voice), 48 kHz float32. The voice is
    mono; the processed take is too, unless a bus has "stereo_echo": then it is (2, samples).

    Timing is settled first: every `stretch` op in the preset's `pre` chain, and
    tempo (1 = as generated, below 1 = slower), are one time-stretch of the clean
    voice before anything else runs. That stretched voice is `voice`: the human
    layer, sample-aligned with the processed take, given the same pitch and reverb
    and the same length, for blending under (or over) the machine.

    blend mixes that voice into the result: 0 = all machine, 0.5 = half and half,
    1 = all voice (see blend_layers). double_ms > 0 lets the machine trail the voice
    by that many ms, a doubling that thickens the blend.

    A dark voice (little energy at 1-4 kHz, where consonants and formants are) is
    brightened before the machine hears it, so its words survive the vocoder (see
    articulation_boost); a preset can opt out with "articulation": false. The voice
    stem stays as recorded.

    bass_db shelves everything under BASS_HZ up or down (negative = less bass), on
    both layers, before loudness normalisation: a take whose words sit low under
    its bass (see speech_lufs) gets clearer with -3 to -9.

    pitch (semitones) applies to the finished sound, after the preset: it moves
    everything down together, the machine's tone and the size of its "throat"
    alike, like a bigger instrument. reverb (a REVERBS key) adds a space last.
    Both layers end at the same loudness, `target_lufs` (see normalise_loudness).

    Empty input gives empty output; silence (and NaN/inf, treated as silence)
    gives silence of the same length rather than amplified noise.
    """
    if reverb not in REVERBS:
        raise ValueError(f"unknown reverb {reverb!r}; choose from {', '.join(REVERBS)}")
    if not 0.0 <= blend <= 1.0 or not 0.0 <= double_ms <= MAX_DOUBLE_MS:
        raise ValueError(f"blend must be 0-1 and double_ms 0-{MAX_DOUBLE_MS:g}")
    if not BASS_RANGE[0] <= bass_db <= BASS_RANGE[1]:
        raise ValueError(f"bass_db must be {BASS_RANGE[0]:g} to {BASS_RANGE[1]:g}")
    voice = timed_voice(audio, sr, preset, tempo, tail_s)
    if not len(voice) or _peak(voice) < SILENCE:
        return np.zeros(len(voice), dtype=np.float32), np.zeros(len(voice), dtype=np.float32)
    pre = [op for op in preset.get("pre", []) if not op.get("off")]
    boost = articulation_boost(voice) if preset.get("articulation", True) else 0.0
    heard = op_eq(voice, bands=[{"type": "highshelf", "hz": 1000.0, "gain_db": boost, "q": 0.7}]) if boost else voice
    x = chain(heard, [op for op in pre if op["op"] != "stretch"])

    # Buses may change length (a stretch inside one): mix at the longest.
    buses = [b for b in (preset.get("buses") or [{"name": "dry", "chain": []}]) if not b.get("off")]
    outs = [(chain(x.copy(), bus.get("chain", [])), db(bus.get("gain_db", 0.0)))
            for bus in buses if not bus.get("stereo_echo")]
    sends = [(chain(x.copy(), bus.get("chain", [])) * db(bus.get("gain_db", 0.0)), bus["stereo_echo"])
             for bus in buses if bus.get("stereo_echo")]
    n = max((len(y) for y, _ in outs), default=len(x))
    mix = np.zeros(n, dtype=np.float32)
    for y, g in outs:
        mix += fit(y, n) * g

    y = np.nan_to_num(chain(mix, preset.get("post", [])))
    voice = fit(voice, len(y))
    if blend:
        y = blend_layers(y, voice, blend, double_ms)
        voice = fit(voice, len(y))
    if bass_db:
        shelf = [{"type": "lowshelf", "hz": BASS_HZ, "gain_db": bass_db}]
        y, voice = op_eq(y, bands=shelf), op_eq(voice, bands=shelf)
    if pitch:
        y = op_pitch(y, semitones=pitch, preserve_formants=False)
        voice = op_pitch(voice, semitones=pitch, preserve_formants=False)
    y = _trim_tail(np.nan_to_num(y))
    voice = op_space(fit(voice, len(y)), reverb)
    y = _trim_tail(np.nan_to_num(op_space(y, reverb)))
    voice = _fade_end(fit(voice, len(y)))
    if sends:
        for send, cfg in sends:
            y = stereo_echo(y, op_pitch(send, semitones=pitch) if pitch else send, **cfg)
        voice = fit(voice, y.shape[-1])
    if _peak(y) < SILENCE:
        return np.zeros_like(y), normalise_loudness(voice, target_lufs)
    return normalise_loudness(y, target_lufs), normalise_loudness(voice, target_lufs)


def stereo_echo(y: np.ndarray, send: np.ndarray, delay_ms: float = 800.0, feedback: float = 0.5,
                lowpass_hz: float = 1100.0, repeats: int = 2, spread: float = 0.8) -> np.ndarray:
    """Bounce `send` behind the take, left and right in turn: repeat k arrives k * delay_ms
    late, darker each time (low-passed at lowpass_hz) and quieter by `feedback`; odd
    repeats lean left, even ones right, by `spread` (0 = centre, 1 = hard). Returns
    (2, samples) stereo, the take itself in the centre."""
    d = int(SR * float(delay_ms) / 1000)
    n = max(y.shape[-1], len(send) + int(repeats) * d)
    out = np.zeros((2, n))
    out[:, : y.shape[-1]] += y
    sos = _lowpass_sos(float(lowpass_hz))
    tap = send.astype(np.float64)
    near, far = (1 + float(spread)) / 2, (1 - float(spread)) / 2
    for k in range(1, int(repeats) + 1):
        tap = signal.sosfilt(sos, np.concatenate([np.zeros(d), tap])) * (1.0 if k == 1 else float(feedback))
        left, right = (near, far) if k % 2 else (far, near)
        out[0, : len(tap)] += left * tap[:n]
        out[1, : len(tap)] += right * tap[:n]
    return _trim_tail(out.astype(np.float32))


def timed_voice(audio: np.ndarray, sr: int, preset: dict, tempo: float = 1.0, tail_s: float = 2.5) -> np.ndarray:
    """The clean voice on a render's timeline: 48 kHz, peak 0.5, stretched by the preset's
    `stretch` ops and tempo, before any effect. Sample 0 is the render's sample 0, so a
    time measured on this is a time in the render (word timings are measured here)."""
    x = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if not len(x):
        return x
    x = to_sr(x, sr)
    if _peak(x) < SILENCE:
        return np.zeros(len(x), dtype=np.float32)
    x /= _peak(x) / 0.5
    x = np.pad(x, (0, int(tail_s * SR)))  # room for reverb tails and delays
    return op_stretch(x, rate=stretch_rate(preset, tempo))


def stretch_rate(preset: dict, tempo: float = 1.0) -> float:
    """The one time-stretch a render applies: the preset's `stretch` ops times tempo.
    A time t in the render is t * stretch_rate in the dry take."""
    pre = [op for op in preset.get("pre", []) if not op.get("off")]
    return float(np.prod([op.get("rate", 1.0) for op in pre if op["op"] == "stretch"])) * tempo


# How bright a voice is: its 1-4 kHz energy against 60 Hz-1 kHz, in dB. The presets
# were tuned on deep male voices measuring -5.5 to -14.6 here, and need nothing. A
# dark voice starves the vocoder of articulation and comes out a muddy drone: an
# "ancient female" at -19 did, and sounded best brightened by +12 dB. Below the floor
# the boost ramps up smoothly (no jump between two takes of one voice) to that maximum.
ARTICULATION_FLOOR_DB = -15.0
ARTICULATION_SLOPE = 3.0
ARTICULATION_MAX_DB = 12.0


def articulation_db(x: np.ndarray) -> float:
    """1-4 kHz energy relative to 60 Hz-1 kHz, in dB (48 kHz audio)."""
    f, p = signal.welch(np.asarray(x, dtype=np.float64), SR, nperseg=8192)
    low, art = p[(f >= 60) & (f < 1000)].sum(), p[(f >= 1000) & (f < 4000)].sum()
    return float(10 * np.log10(max(art, 1e-20) / max(low, 1e-20)))


def articulation_boost(x: np.ndarray) -> float:
    """How many dB a high shelf at 1 kHz should lift this voice for the presets (0 if bright enough)."""
    if _peak(x) < SILENCE:
        return 0.0
    gain = ARTICULATION_SLOPE * (ARTICULATION_FLOOR_DB - articulation_db(x))
    return round(float(np.clip(gain, 0.0, ARTICULATION_MAX_DB)), 1)


MAX_DOUBLE_MS = 60.0
BASS_HZ = 150.0
BASS_RANGE = (-12.0, 6.0)

# Where the words are: 300 Hz-4 kHz. Measured on the finished -16 LUFS take, its
# loudness says whether the words carry or sit under the bass. Calibrated by ear on
# the reaper-drone takes of a 39-line film: clear takes measured -20.6 to -23.4,
# borderline ones -23 to -24.4, and the one judged unintelligible -28.
SPEECH_BAND = (300.0, 4000.0)
SPEECH_QUIET_LUFS = -25.5     # below this: warn, suggest less bass
SPEECH_BURIED_LUFS = -27.0    # below this: the base voice itself is too boomy; another seed


def speech_lufs(y: np.ndarray) -> float:
    """Loudness (LUFS) of the 300 Hz-4 kHz band alone: how loud the words are in a take.
    -inf for silence, or a take too short to filter (a few ms)."""
    if y.shape[-1] < 64:
        return float("-inf")
    sos = signal.butter(4, [SPEECH_BAND[0] / (SR / 2), SPEECH_BAND[1] / (SR / 2)], "band", output="sos")
    return loudness(signal.sosfiltfilt(sos, y.astype(np.float64), axis=-1).astype(np.float32))


def blend_layers(machine: np.ndarray, voice: np.ndarray, blend: float, double_ms: float = 0.0) -> np.ndarray:
    """Equal-power mix of the machine and the voice under it, loudness-matched first.

    blend is the voice's share: 0 = all machine, 1 = all voice; 0.5 gives both
    the same loudness. The two are effectively uncorrelated (the machine's tone
    is synthetic), so cos/sin gains keep the total loudness steady across the range.
    double_ms delays the machine by that much: 10-30 ms reads as one thicker,
    doubled voice rather than an echo.
    """
    def matched(x):
        level = loudness(x)
        return x * db(-23.0 - level) if np.isfinite(level) else x

    m, v = matched(machine), matched(voice)
    d = int(SR * double_ms / 1000)
    if d:
        m, v = np.pad(m, (d, 0)), np.pad(v, (0, d))
    th = float(blend) * np.pi / 2
    return (np.cos(th) * m + np.sin(th) * v).astype(np.float32)


TARGET_LUFS = -16.0   # speech on phones, podcasts, messengers
PEAK_CEILING_DB = -1.0


def normalise_loudness(y: np.ndarray, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Bring a take to `target_lufs` integrated loudness, peaks held under -1 dBFS.

    Peak normalisation alone makes every take touch -1 dBFS, but a dense, distorted
    preset then sounds several dB louder than a dynamic one. This matches what the
    ear hears instead. When the gain pushes peaks over the ceiling, a limiter
    catches them; it costs a little loudness, so the gain is corrected and the
    limiter re-run (twice is enough: the correction is a fraction of a dB).
    """
    y = y.astype(np.float32)
    current = loudness(y)
    if not np.isfinite(current):
        return y
    out = y * db(target_lufs - current)
    for _ in range(3):
        if _peak(out) > db(PEAK_CEILING_DB):
            out = run([pb.Limiter(threshold_db=PEAK_CEILING_DB - 0.5, release_ms=80)], out)
        miss = target_lufs - loudness(out)
        if abs(miss) < 0.2:
            break
        out = out * db(miss)
    # Belt and braces: never leave a sample above the ceiling.
    if _peak(out) > db(PEAK_CEILING_DB):
        out = out * (db(PEAK_CEILING_DB) / _peak(out))
    return out.astype(np.float32)


# ITU-R BS.1770 K-weighting at 48 kHz: a high shelf (the head's acoustic effect) then
# a high-pass (the "RLB" curve). Coefficients from the standard.
_K_SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
_K_HP = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def loudness(x: np.ndarray) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4, mono or (2, samples) stereo, 48 kHz),
    the way streaming services and broadcasters measure it: K-weighted, 400 ms blocks,
    channel powers summed, gated so silence and quiet tails don't drag it down. -inf
    for silence."""
    y = signal.lfilter(*_K_HP, signal.lfilter(*_K_SHELF, x.astype(np.float64), axis=-1), axis=-1)
    y = y.reshape(-1, y.shape[-1])  # (channels, samples)
    block, step = int(0.4 * SR), int(0.1 * SR)
    n = y.shape[-1]
    if n < block:  # too short to gate: plain K-weighted mean square
        ms = np.array([np.mean(y ** 2, axis=-1).sum()])
    else:
        ms = np.array([np.mean(y[:, i:i + block] ** 2, axis=-1).sum() for i in range(0, n - block + 1, step)])
    lufs = -0.691 + 10 * np.log10(ms + 1e-20)
    keep = lufs > -70.0                                   # absolute gate
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10 * np.log10(ms[keep].mean()) - 10.0  # relative gate
    keep &= lufs > rel
    return float(-0.691 + 10 * np.log10(ms[keep].mean()))


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _trim_tail(x: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Cut trailing near-silence left by the tail padding, keeping a short fade (mono or stereo)."""
    thresh = _peak(x) * db(floor_db)
    env = np.abs(x).max(axis=0) if x.ndim == 2 else np.abs(x)
    loud = np.nonzero(env > thresh)[0]
    if not len(loud):
        return x
    end = min(x.shape[-1], loud[-1] + int(0.05 * SR))
    return _fade_end(x[..., :end])


def _fade_end(x: np.ndarray, seconds: float = 0.05) -> np.ndarray:
    y = x.copy()
    fade = min(y.shape[-1], int(seconds * SR))
    if fade:
        y[..., -fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return y


def mono(y: np.ndarray) -> np.ndarray:
    """A stereo take folded to mono (for transcripts and pictures); mono passes through."""
    return y.mean(axis=0).astype(np.float32) if y.ndim == 2 else y
