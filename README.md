# local-voice

A private, fully local text-to-speech service for **non-human voices**, in **English and
Romanian**, built for AI agents (and people) to call. Describe a voice and a mood, give it a
line, and get back a colossal machine intelligence: Mass Effect's Reapers are the reference.

It runs **Breeze TTS 2** (English; the top open-weights voice-design TTS model as of autumn
2026) and **OmniVoice** (Romanian, with correct diacritics and numbers) on your own GPU, then
turns the voice into a machine with a vocoder-based sound-design chain. No accounts, no cloud:
text and audio never leave the machine.

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1   # ~23 GB: weights + two PyTorch venvs
.\start.cmd                                              # opens http://127.0.0.1:8190/
```

## For AI agents (Claude Code skills)

Two Claude Code skills come with the repo:

| Skill | What it's for |
|---|---|
| `local-voice` (`SKILL.md`) | making audio: lines, scenes and voices, judged with Whisper transcripts and spectrograms, delivered as files |
| `speak` (`skills/speak/SKILL.md`) | **talking to you out loud**: the agent says short replies on your speakers (and in chat) when you ask it to speak, "tell me when it's done", "read this to me" |

Install both as links in your personal skills folder (links, not copies, so they follow the repo):

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE\.claude\skills\local-voice" -Target (Resolve-Path .).Path
New-Item -ItemType Junction -Path "$env:USERPROFILE\.claude\skills\speak" -Target (Resolve-Path .\skills\speak).Path
```

The `speak` skill finds the tool at `~/.claude/skills/local-voice`, so install `local-voice` at
exactly that path. Then, in Claude Code, ask something like "say hello out loud" or "tell me by
voice when the build finishes". The server must be running (`start.cmd`) or the agent starts it.

Any other agent can use the CLI (`scripts/say.py`, standard library only) or the HTTP API below.
The essentials:

```powershell
python scripts\say.py "We are the dark."                      # a take with the defaults
python scripts\say.py "We are the dark." --play --no-check --no-save   # say it out loud, keep nothing
python scripts\say.py "Noi suntem întunericul." --lang ro --play
python scripts\say.py --settings                              # which defaults are in effect
```

## How it works

No open model can *speak* as a Reaper: every TTS model is trained on human speech. So there
are two stages:

1. **Voice design (the model).** Breeze TTS 2 builds a voice from a description such as
   *"Ancient male, massive scale. Deep, resonant bass, gravelly, subterranean. Speaks very slowly
   and calmly with cold, absolute authority."* No reference audio is involved. That prompt lands
   around 65-70 Hz, a very low bass, and the delivery (pace, pauses, emphasis) comes from here.
2. **Sound design (the presets).** The human voice is *replaced* by a machine. A 36-band channel
   vocoder lays the voice's articulation over a synthetic brass tone: a detuned sawtooth stack
   with inharmonic "scraped metal" partials and a sub-octave, at a fixed low pitch (F1, 43.65 Hz)
   that follows the voice's intonation only as much as the preset says. When the voice hisses
   (s, sh, f, t) the carrier crossfades to noise, which keeps consonants intelligible. Then
   brassy distortion, EQ (the drone's low end cut 6 dB below 150 Hz, so it doesn't boom over the
   words) and a metal-tube resonance. The presets are **dry** (no reverb or echo):
   a clean, close voice.
3. **Finishing (optional, any preset).** `tempo` (time-stretch, same pitch) is applied first,
   together with the preset's own slow-down, as one stretch of the clean voice. `pitch`
   (semitones; negative is lower and bigger: the machine's tone and its "throat" move down
   together) and `reverb` (`none`, `room`, `hall`, `cathedral`) are applied to the finished
   sound. Re-running them on an existing take takes a second or two.
4. **The voice stem.** Every render also writes `..._voice.wav`: the human voice the machine was
   driven by, sample-aligned with the processed take, with the same tempo, pitch, reverb, length
   and loudness. Lay it under the processed track to mix machine and human, or let the tool do it:
   `blend` (0-1, the human's share, equal-power and loudness-matched) and optionally `double_ms`
   (the machine trails the voice by 10-30 ms for a thicker, doubled texture). (The vocoder filters
   forwards and backwards so the machine speaks exactly in time with the voice; one-pass filters
   made it 10-90 ms late depending on pitch, which sounded doubled when blended.)
5. **Loudness.** Every take ends normalised to **-16 LUFS** (ITU-R BS.1770 integrated loudness,
   the usual target for speech on phones and podcasts) with peaks held under -1 dBFS, so all
   presets, pitches and reverbs play at the same perceived volume.

Pitch-shifting and distorting a human voice instead gives you a Doom-style demon: organic, not
mechanical. That version is kept as the `demon` preset for comparison.

### Two engines, one voice

| Language | Engine | How the voice is chosen |
|---|---|---|
| English (`en`) | Breeze TTS 2 | a free-text description (`instruction`), strength `cfg_scale` |
| Romanian (`ro`) | OmniVoice | by default it **clones the default voice** (below); or fixed attributes, comma-separated, e.g. `male, elderly, very low pitch`; the model's pace via `speed` (0.75-1.3, default 0.9) |

OmniVoice needs a newer `transformers` than Breeze allows, so it lives in its own venv
(`.venv-ro`) and the server runs it as a child process (`server/omni_worker.py`), started on the
first Romanian request. Romanian text is normalised first: numbers, decimals and clock times
are spelled out ("2026" -> "două mii douăzeci și șase", "3,5" -> "trei virgulă cinci", "10:30"
-> "zece și treizeci") and legacy cedilla letters (ş ţ) become the correct comma-below ones (ș ț).

**Romanian is generated a sentence at a time and checked.** OmniVoice drops words on some seeds,
more often the longer the input: a 37-word paragraph lost up to 7 words on one seed in three.
So each sentence is generated separately, transcribed with Whisper, and regenerated on a new
seed (up to 3 tries) if words are missing; the best try is kept and the sentences are joined
with a short pause. The same paragraph then came out complete on every seed tried. A take whose
sentence still misses words after 3 tries carries a warning. Below `speed` 0.75 the model starts
repeating words to fill the time it's given, so the slowest settings are `tempo`'s job.

**The default voice.** OmniVoice's own deepest design ("very low pitch") lands around 100-130 Hz,
which sounds small under the Reaper presets. Breeze's English Reaper voice sits near 65 Hz. So
by default Romanian lines **clone the default voice**: an English take. If there isn't one, the
first Romanian line makes it with Breeze, trying its own seeds until the voice lands under 75 Hz
(seed 12 gives ~65 Hz; unrelated to the default seed for takes), and stores a copy in `voices\` (safe from clearing `output\`). Cloned Romanian keeps that depth
(66-69 Hz measured) and Whisper still transcribes it perfectly. Set your own with **Set as
default voice** on any take, or `say.py --default-voice <take>`. Whisper can't judge accent, so
listen for English colouring in cloned Romanian.

**Clone voice for next lines / `--like` works across engines and languages** too: any take's voice can
speak either language. In English, `--like` without `--instruction` is a pure clone; with one,
Breeze keeps the voice but steers its delivery ("voice direction").

### Presets

| Preset | Character |
|---|---|
| `sovereign` | Tuned to the measured pitch structure of a reference clip (though `reaper-drone` sounded closer by ear): the human voice stays audible (~66 Hz, with its intonation), and a buzzing hum an octave below (C1, 33 Hz) sounds only while it speaks and rings briefly after each phrase. |
| `reaper-drone` | The brass machine tone does all the speaking at one unmoving pitch: a colossal drone shaped only by the words. The most electronic, and judged closest to the Reapers by ear. **Default.** |
| `reaper` | Like `reaper-drone`, but loosely following the voice's intonation. |
| `harbinger` | A machine imitating speech: a synth chord (root, fifth, octave) follows the intonation but snaps to whole semitones, with ring-mod shimmer and heavy distortion. |
| `demon` | The human voice pitched down and distorted, with octave and sub layers: monstrous rather than mechanical. |
| `dry` | No effects, for judging the base voice. |

All of them transcribe at 100% of words with Whisper on four test takes (`demon`: 88% on one).

**Dark voices are brightened for the machine.** The presets were tuned on deep male voices,
which have plenty of energy at 1-4 kHz, where consonants and formants are. A darker voice (a
low female one, say) starves the vocoder there and comes out a muddy drone, so before the
machine hears it, a voice measuring below -15 dB on that band (relative to 60 Hz-1 kHz) is
lifted with a high shelf at 1 kHz, up to +12 dB. Male voices measure -5.5 to -14.6 and are left
alone; the voice stem and the `dry` preset are never changed. The take reports
`articulation_boost_db`; a preset can opt out with `"articulation": false`.

Add your own by copying a JSON file in `voicefx/presets/`. A preset is a `pre` chain, parallel
`buses` mixed at `gain_db`, and a `post` chain. Every op and its parameters are listed in
`voicefx/fx.py` (`OPS`, `op_vocoder`). The vocoder's main knobs are `f0` (the machine's pitch),
`follow` (0 = monotone, 1 = full intonation), `steps` (snap to semitones), `chord`, `metal` and
`sub`. Presets are re-read on every request, and re-running effects on an existing take takes a
second or two with no regeneration. Name scratch presets `_something.json` and git ignores them.

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 |
| GPU | NVIDIA, 16 GB VRAM for everything at once (Breeze + fast paths ~8.4 GB, Whisper ~1.5 GB, OmniVoice ~2.7 GB: ~12.6 GB). English only fits in 12 GB. RTX 50-series supported (CUDA 12.8 wheels) |
| Disk | ~23 GB (weights 12.6 GB, two venvs ~10 GB) plus pip cache |
| Software | Python 3.10+ (3.13 by default, via the `py` launcher), Git |

## Install

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

It clones the Breeze TTS 2 inference code at a pinned commit into `engine\`, builds `.venv\` with
PyTorch 2.9.1 + CUDA 12.8, installs `triton-windows` for the fast path, builds `.venv-ro\` for
OmniVoice (pinned commit, `requirements-ro.txt`), and downloads the weights into `models\`. Re-running is safe: finished steps are skipped and downloads resume.

**Weights are pinned:** `models.json` fixes each HuggingFace repo to a revision and every file to
a SHA256, so an upstream re-upload fails loudly instead of silently changing what you get.

| Model | Size | Role | Licence |
|---|---|---|---|
| [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | 7.7 GB | voice-design TTS (24 kHz) | Research and non-commercial |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | 3.3 GB | Romanian TTS (24 kHz), voice design by attributes, cloning | CC-BY-NC |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 1.6 GB | transcribes takes (English and Romanian) to check intelligibility | MIT |

Options: `-SkipModels`, `-ReuseModels <other install>\models` (hard-link existing weights instead of downloading), `-PythonVersion 3.12`, `-CudaIndex cu130`, `-NoPipCache`.

Verify the weights at any time without downloading anything:
`.venv\Scripts\python.exe scripts\download_models.py --verify`

## Using it

**`.\start.cmd`** loads the model (~10 s), warms up the CUDA-graph fast paths (~35-50 s), then
opens the page. It always runs in its own console window titled "local-voice server - close this
window to stop it", and `start.cmd` returns at once, so a server is never running out of sight;
closing the window stops it and frees the GPU. `start.cmd --no-browser` does the same without
opening the page (that's how agents start it); `start.cmd --here` runs it in the current console
instead, for scripts that manage the process themselves. A second start
while one is running exits at once ("port in use") instead of loading another copy. Close the
console window, or run `.\stop.cmd`, to free the VRAM: it stops the server recorded in
`output\.server.pid` and its Romanian worker, however it was started.

**Standby.** After 15 minutes with no takes the server unloads its models by itself and keeps
running with almost nothing on the GPU (~0.2 GB), so another tool can have the card; the next
request loads them again (~20 s once the fast paths have been compiled before, ~1 min the
first time). `say.py --unload` (API: `POST /api/unload`) does it at once. Change the wait with
`server\app.py --idle-unload MIN` (0 = never).

### Command line (what agents use)

**Defaults.** With nothing but the text, you get the default voice: the `reaper-drone` preset at
strength 6 (`--cfg 6`), seed `74525861` (a voice picked by ear), Romanian speed 0.9, a light room (`reverb room`), pitch 0,
tempo 1, and the Whisper check on. The page starts with the same settings. Override any of them
per request; `--seed random` (API: `"seed": null`) for a fresh voice, `--reverb none` for a dry
take, `--no-check` (API: `"check": false`) to skip the check.

```powershell
python scripts\say.py "Your civilization is a flicker in the dark." --check --out D:\clips
python scripts\say.py "We are the end of everything." --preset harbinger --seed 7
python scripts\say.py "And the beginning." --like 20260101-120000-ab12      # same voice
python scripts\say.py --fx 20260101-120000-ab12 --pitch -5 --reverb hall     # effects only
python scripts\say.py "We are the dark." --tempo 0.85 --format mp3 --out D:\clips  # for WhatsApp
python scripts\say.py "Noi suntem întunericul." --lang ro                  # default voice
python scripts\say.py "Noi suntem întunericul." --lang ro --like 20260101-120000-ab12
python scripts\say.py "Noi suntem întunericul." --lang ro --instruction "male, middle-aged, low pitch" --speed 0.8
python scripts\say.py --default-voice 20260101-120000-ab12                    # set the default voice
python scripts\say.py --presets
```

| Flag | Meaning |
|---|---|
| `--preset NAME` | Effects preset (default `reaper-drone`). |
| `--lang CODE` | `en` (default): Breeze. `ro`: OmniVoice. |
| `--instruction "..."` | The voice description. English: free text, default the preset's suggested one, `""` for none. Romanian: attributes such as `male, elderly, very low pitch`; omitted means clone the default voice. |
| `--seed N` | Reproduces a take exactly. Default: the default seed, 74525861. `random` for a new voice. |
| `--cfg X` | English only: how hard the model follows the instruction (default 6; Breeze recommends 4; range 0.5-10). |
| `--speed X` | Romanian only: the model's speaking rate, 0.75-1.3 (default 0.9). Slower than that: `--tempo`. |
| `--pitch N` | Semitones, -12..12, on the finished sound (e.g. -4 for lower and bigger). |
| `--tempo X` | Time-stretch, 0.5-1.5 (0.85 = slower, same pitch). |
| `--blend X` | Mix the human voice under the machine, 0-1: 0 all machine, 0.5 half and half (equal loudness), 1 all human. Sample-aligned. |
| `--double MS` | With `--blend`: the machine trails the voice by MS ms (0-60). 10-30 thickens the two into one doubled voice (it can read as extra reverb); 0, the default, keeps them exactly together. |
| `--bass DB` | Shelve everything under 150 Hz, -12..6 dB, before loudness matching. -6 when a take warns that the words sit low under the bass. |
| `--trim-tail` / `--no-trim-tail` | Cut stray sound after the last scripted word (an "mmm", a breath), in the gap after it; the take remembers the cut, so later `--fx` renders keep it. The dry file is untouched; `--no-trim-tail` undoes it. |
| `--reverb KIND` | `none`, `room` (light; default), `hall` (medium), `cathedral` (heavy, ~4.5 s tail). |
| `--like TAKE` | Keep an earlier take's voice for a new line, in either language, so a scene sounds like one speaker. |
| `--default-voice [TAKE]` | Show the default voice, set it to TAKE, or `previous` to undo the last change. |
| `--settings` | Show the defaults in effect (built-in, or saved in `config.local.json`). |
| `--fx TAKE` | Re-run effects (preset, pitch, tempo, reverb) on an earlier take; no new generation. |
| `--check` / `--no-check` | Whisper transcribes dry and processed audio, prints a words-heard score, and draws a spectrogram. On by default. |
| `--out DIR` / `--name STEM` | Copy the processed audio, its voice stem and the dry WAV to DIR (never overwrites). |
| `--format F` | With `--out`: `both` (default: processed WAV + MP3, voice stem WAV and dry WAV), `wav` (the same without the MP3) or `mp3` (just the MP3: WhatsApp and most messengers take MP3, not WAV). |
| `--wait S` | Wait up to S seconds for the server (startup takes ~60 s). |
| `--json` | Print only the server's JSON response, plus `paths` (the delivered files). |
| `--play` | Play the processed take on this machine's speakers; returns when playback ends (exit 1 if it couldn't play). |
| `--unload` | Free the GPU now: the server drops its models but keeps running (it also does this after 15 idle minutes). |
| `--no-save` | Delete what this call made from `output\` when done (after `--play` and/or `--out`): the new take, or with `--fx` only that preview's files. For speech meant only to be heard; the `speak` skill uses it. |

Exit codes: 0 done, 1 no server / lost the server / server error, 2 bad arguments.

**Your own defaults.** On the page, **Save as my defaults** stores the panel's settings in
`config.local.json` (local, gitignored); **Reset to built-in** deletes it. Both ask first. Saved
settings override the built-in ones for the page and for every API/CLI request that leaves a
setting out (the language is never switched for an agent). `say.py --settings` shows what's in
effect and where each value comes from.

**The server's log.** The window shows milestones, one line per take and any errors; the
full output (warmup steps, library notices) goes to `logs\server.log`, timestamped. Each start
keeps the previous run as `logs\server.previous.log`, so a crash's details survive a restart.

**What's safe to delete.** `output\` holds takes only: clear it any time. `logs\` too. The default voice is a
copy of its recording in `voices\` (with the previous one, for Undo), and your saved settings are
in `config.local.json`; neither lives in `output\`. All of these are gitignored.

### HTTP API (127.0.0.1:8190)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/speak` | `{"text", "language"?, "instruction"?, "preset"?, "seed"?, "cfg_scale"?, "speed"?, "pitch"?, "tempo"?, "blend"?, "double_ms"?, "bass_db"?, "trim_tail"?, "reverb"?, "like"?, "check"?}` | the take: id, seed, `files` (dry, wet, mp3, voice; with `check` spectrogram, words, srt), `words` (word timings), durations, timings; `heard` / `match` / `asr_unreliable` with `check` (or `check_error`); Romanian: per-`sentences` scores; `warnings` |
| `POST /api/fx` | `{"id", "preset"?, "pitch"?, "tempo"?, "blend"?, "double_ms"?, "bass_db"?, "trim_tail"?, "reverb"?, "check"?}` | the take re-processed |
| `GET /api/presets` | | names, descriptions, suggested voice descriptions |
| `GET /api/health` | | which models are loaded, `standby`, idle seconds, VRAM |
| `POST /api/unload` | | free the GPU: drop every model, keep running (standby) |
| `GET` / `POST /api/default-voice` | `{"id"}` to set | the default voice: source take id, text, median pitch, `previous` |
| `POST /api/default-voice/restore-previous` | | swap back to the previous default voice |
| `GET` / `POST` / `DELETE /api/settings` | the settings to save | `effective`, `saved`, `repo` defaults; save / reset `config.local.json` |
| `DELETE /api/takes/<id>` | | delete every file of one take (what `--no-save` does) |
| `DELETE /files/<take>_fx-...` | | delete one rendered file (what `--fx --no-save` does); a take's dry voice and record can't be deleted this way |
| `GET /files/<name>` | | a WAV, MP3 or PNG from `output/` |

`instruction` omitted means: English, the preset's suggested description; Romanian, clone the
default voice (made on first use). In English `""` means none (plain TTS, or a pure clone with
`like`). Bad input (unknown language, preset or reverb, a seed outside 0..2^32-1, blank text)
is a 4xx with a message, not a 500. Requests must use Host `127.0.0.1` or `localhost`.

**Length:** text is capped at 800 characters, and Breeze stops at 120 s of audio: a take that hits
the ceiling carries a warning that its end is probably missing. Split long scripts into several
takes and keep the voice with `like`.

### Takes

Every generation is a *take*, with an id like `20260101-120000-ab12`. Its files are in `output\`:

| File | What |
|---|---|
| `<id>_dry.wav` | the voice as generated, 24 kHz (Breeze or OmniVoice) |
| `<id>_fx-<preset>[_pitch-4][_tempo0.85][_hall].wav` / `.mp3` | processed, 48 kHz; one pair per combination of effect settings |
| `<id>_fx-...._voice.wav` | the human voice under that render: sample-aligned, same tempo, pitch, reverb, length and loudness |
| `<id>_fx-...._words.json` / `.srt` | (with `check`) word timings for that render, and its phrases as subtitles |
| `<id>.json` | text, language, model, instruction, seed, cfg or speed: everything needed to redo it |
| `<id>_fx-....png` | waveform and spectrogram (with `check`) |

### Warnings a take can carry

- **The words sit low under the bass** (every render): the 300 Hz-4 kHz band, where the words
  are, measured on the finished take. Clear takes measure about -21 to -23 LUFS; below -25.5 the
  warning suggests `--bass -6`, below -27 another seed (a take that boomy didn't recover with EQ).
- **Sound after the last word / before the first** (with `check`): sound outside the script,
  found in the audio itself (Whisper often doesn't write an "mmm" down), labelled voiced (a hum,
  an "mmm") or breath/noise. `--trim-tail` cuts a trailing one.
- **Words heard that aren't in the script** (with `check`).

### Word timings

With `check` on, every render also gets `..._words.json`: when each word of the script starts and
ends in that render, in seconds, for cueing animation to the words. It's measured on the clean
voice on the render's own timeline (so tempo and the presets' slow-down are already in it):
Whisper's word times, matched to the script (numbers spelled out), then each boundary moved into
the pause between two words where there is one, and each edge snapped to where the sound starts
or stops. Across a pause that's accurate to about 10 ms; between words spoken together it's
Whisper's own boundary, usually within a few tens of ms. A word Whisper misheard or missed still
gets a time, marked `"approx": true`. The same file has `phrases` (split at punctuation and
pauses) and `extras` (anything heard that isn't in the script); `.srt` has the phrases as
subtitles for an editor's timeline. `--out` delivers both as `<name>_words.json` and `<name>.srt`.

### The page

The page at http://127.0.0.1:8190/ does the same: text, language, preset (which loads its
suggested voice description), voice description, seed, strength (English) or speed (Romanian),
pitch, tempo, reverb, **Transcribe + spectrogram**, and per take: **WAV** / **MP3** / **voice WAV** download
links, an **Effects** box (another preset, pitch, tempo or reverb on the same take, no
regeneration), **Clone voice for next lines** (later lines clone it, in either language), **Set as default
voice** (what Romanian clones when the voice attributes are left empty) and **Load settings**
(copy the take's settings back into the panel). Hover over any control for 1 s for an
explanation. `Ctrl+Enter` speaks.

### Speed (RTX 5080, warm)

| | |
|---|---|
| Generation, English | ~2-2.5x faster than real time (a 12 s line in ~5 s) |
| Generation, Romanian | ~1-1.5 s a sentence including its Whisper check; ~4 s for a 3-sentence paragraph. The first after a start ~20-30 s (starts the worker, may make the default voice) |
| Effects | ~0.2x real time |
| Whisper check | ~1 s |

Without the fast paths (`start.cmd --safe`) generation is about 2.5x *slower* than real
time.

## Notes and gotchas

- **The base voice varies by seed.** Most seeds land very deep (65-70 Hz). Some come out around
  95 Hz, which sounds less huge. Reroll.
- **Whisper is a rough proxy.** On heavily processed audio it can loop ("oh, oh, oh...") or swing
  between 0% and 90% on near-identical inputs. Loops are flagged as unreliable. Judge presets
  across several takes, and trust your ears.
- **Windows-specific fixes** in `server/app.py`: the depth-decoder fast path compiles with
  Inductor, whose static CUDA launcher overflows a 32-bit C `long` on Windows, so it is disabled
  (`TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0`). FlashAttention isn't needed: SDPA is used.
- The engine prints `SoX could not be found` and a flash-attn warning at start. Both are harmless.

## Privacy

The server binds to 127.0.0.1 only (hard-coded) and forces HuggingFace and Gradio offline, so
after installation nothing contacts the network. `output\` holds every take (gitignored); delete
it whenever you like.

## Before pushing

```powershell
python scripts\check_repo_safe.py            # what git would publish
python scripts\check_repo_safe.py --history  # and every past commit
```

It fails on weights, audio, images, large files, local paths, emails, IP addresses and
credential-shaped strings. It can't judge prose, so read your own diff too.

## Licensing

This repo's code is MIT. **The Breeze TTS 2 and OmniVoice weights, and everything generated
with them, are for research and non-commercial use only.** See [LICENSE-MODELS.md](LICENSE-MODELS.md). Mass Effect
and the Reapers belong to EA / BioWare. This tool imitates a processing style and contains no
game audio.

## Repo layout

```
install.ps1 / start.cmd / stop.cmd   setup, launch, shutdown
models.json                          pinned, checksummed weight manifest
requirements.txt, requirements-ro.txt  the two venvs' dependencies
server/app.py, server/page.html      the loopback server and its one-page UI
server/omni_worker.py, textnorm.py   the Romanian engine (run in .venv-ro), its text normaliser
voicefx/fx.py, presets/              the effects engine and presets
scripts/say.py                       CLI (standard library only)
scripts/download_models.py           resumable, verifying downloader
scripts/check_repo_safe.py           pre-publish scan: no weights, audio, local paths or secrets
tests/                               effects and text tests (pytest)
SKILL.md, skills/speak/SKILL.md      Claude Code skills: make audio / talk out loud
engine/, models/, .venv*/, output/, logs/, voices/   created at install / run time, gitignored
```
