---
name: local-voice
description: Generate speech locally in English (Breeze TTS 2 voice design: describe the voice and mood in words) or Romanian (OmniVoice, correct diacritics) on this machine's GPU, then turn it into a machine with vocoder-based sound-design presets (Mass Effect Reaper / Harbinger style), check every take with Whisper and a spectrogram, and deliver the keepers to a folder the user names. Use when the user asks for local TTS, local voice generation, a Reaper or other non-human / monster / machine voice, or wants speech made without a cloud service.
---

# local-voice: voice-design TTS + non-human effects on the local GPU

The model makes a **human** voice: Breeze TTS 2 (English) from a written description like
"ancient male, massive scale, deep resonant bass, subterranean, slow and cold"; OmniVoice
(Romanian) by cloning the "house voice" (a deep English take) by default, or from attributes like
"male, elderly, very low pitch". No open model makes a truly
non-human voice by itself, so the presets replace the voice with a machine: a vocoder lays its
articulation over a synthetic brass tone at a fixed low pitch, then distortion, metal resonance
and EQ. The presets add no reverb or echo: clean, close voices, with the room added later. The
delivery (pace, pauses, emphasis) still comes from the model.

**You cannot hear the audio.** You judge takes from Whisper transcripts (did the words survive the
effects?), spectrograms (is the energy where the preset intends?) and the numbers. The user is the
only judge of how it *sounds*: send them the files early and let them steer.

## How a request looks

"Make the Reaper say: *Your extinction is not a tragedy. It is a correction.*"

1. **Check** the GPU and the server (step 1). Start the server if needed.
2. **Draft:** start with the house defaults (just the text); for alternatives, add
   `--seed random` for 2-3 takes (step 2).
3. **Read** the transcripts: wet below ~85% of words means the effects are burying the speech.
   Open the spectrogram PNG with Read if something looks odd.
4. **Iterate:** another seed (the base voice varies a lot between seeds), a different
   `--instruction`, another preset, or `--fx` to re-run effects on a take without regenerating.
5. **Deliver** the best with `--out` to where the user wants it, and send them the files
   (SendUserFile) so they can listen.
6. **Report:** paths, take ids and seeds, the instruction used, the words-heard scores, and that the
   server is still holding VRAM (step 5).

**When to ask the user:**
- Before stopping the local-agent model server (`llama-server.exe`) or the local-image ComfyUI
  server: they may be in use.
- Whether a take sounds right: you can't tell. Offer concrete knobs ("more metal", "slower",
  "a steadier pitch", "deeper") rather than asking open-endedly.
- Before writing audio into their project, if they didn't say where it goes.

All paths below are relative to this skill's base directory (the folder containing this file).
`README.md` has the full details.

## Ground rules

- **Everything stays local.** The server listens on 127.0.0.1 only. Never send the user's text
  or audio to another service, and don't fall back to a cloud TTS without asking.
- **Licence:** Breeze TTS 2's and OmniVoice's weights *and the audio they produce* are research
  and non-commercial only (`LICENSE-MODELS.md`). If the user plans to
  sell or publish something commercially with it, mention the licence once.
- **Use your own judgment on content**, as if you were speaking it yourself. No voice clones of
  real people (this setup only does voice design and re-using its own earlier takes), and nothing
  made to deceive someone about who is speaking.
- **Languages:** English (`--lang en`, default) goes to Breeze; Romanian (`--lang ro`) to
  OmniVoice. No other languages are supported.

## 1. Check the GPU and the server

```bash
curl -s http://127.0.0.1:8190/api/health      # JSON means the server is up
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
tasklist | grep -i -E "llama-server|python"
```

**The 16 GB card holds one model stack at a time.** This server needs ~8.4 GB for English,
~10 GB with Whisper, and ~12.6 GB once Romanian has been used (the OmniVoice worker adds ~2.7 GB). The local-agent server (`llama-server.exe`, ~11-12 GB) and the local-image ComfyUI
server (~15.5 GB) can't run alongside it. If one is up, **ask the user** before stopping it with
that skill's own `stop.cmd`.

**Start the server** in the background (run_in_background), without opening a browser:

```bash
cmd //c "<base dir>\\start.cmd" --no-browser
```

It loads the model in ~10 s, then **warms up its fast paths for ~35-50 s** before it answers. The
`--wait` flag below covers that, so you can start it and generate right away with `--wait 180`.

If the fast-path warmup fails (a Triton or driver problem), run the server directly with `--safe`:
eager mode, about 6x slower but no compilation:
`.venv\Scripts\python.exe server\app.py --safe`.

**If `.venv` or `models/` is missing,** the tool isn't installed. `install.ps1` downloads ~23 GB;
ask the user before running it.

## 2. Commands

Run the CLI with any Python 3 (standard library only):

```bash
python "<base dir>/scripts/say.py" "<text>" --check --wait 180 --out <dir>
python "<base dir>/scripts/say.py" "<text>" --preset harbinger --seed 7 --check --out <dir>
python "<base dir>/scripts/say.py" "<text>" --instruction "<voice description>" --check --out <dir>
python "<base dir>/scripts/say.py" "<text>" --like <take id> --check --out <dir>
python "<base dir>/scripts/say.py" --fx <take id> --preset harbinger --pitch -4 --reverb hall --out <dir>
python "<base dir>/scripts/say.py" "<romanian text>" --lang ro --check --out <dir>
python "<base dir>/scripts/say.py" "<romanian text>" --lang ro --like <take id> --tempo 0.85 --check --out <dir>
python "<base dir>/scripts/say.py" "<text>" --format mp3 --out <dir>     # for WhatsApp / messengers
python "<base dir>/scripts/say.py" --presets
```

**House defaults.** With nothing but the text, you get the house voice: the `harbinger` preset,
seed `74525860` (a voice picked by ear), Romanian speed 0.9, a light room (`reverb room`), pitch 0,
tempo 1, and the Whisper check on. The page starts with the same settings. Override any of them
per request; `--seed random` (API: `"seed": null`) for a fresh voice, `--reverb none` for a dry
take, `--no-check` (API: `"check": false`) to skip the check.

| Flag | Meaning |
|---|---|
| `--preset NAME` | Effects preset (default `harbinger`). `--presets` lists them. See the table below. |
| `--lang CODE` | `en` (default): Breeze. `ro`: OmniVoice. |
| `--instruction "..."` | The voice description. English: free text, default the preset's suggested one (with `--like`, default is a pure clone; an explicit description steers the kept voice's delivery). Romanian: omit it to clone the house voice (the default, and the deep one); or comma-separated attributes, one per group: gender `male`/`female`; age `child`, `teenager`, `young adult`, `middle-aged`, `elderly`; pitch `very low pitch` ... `very high pitch`; style `whisper`. Unknown attributes are an error. |
| `--seed N` | Reproduces a take exactly. Default: the house seed `74525860`; `random` for a new voice (do that when the user wants alternatives). Printed every time; note it for keepers. |
| `--cfg X` | English only: how hard the model follows the instruction (default 4, Breeze's recommendation; 0.5-10, above ~5-6 tends to sound forced). |
| `--speed X` | Romanian only: the model's speaking rate, 0.75-1.3 (default 0.9). Below 0.75 it repeats words, so the server refuses it: use `--tempo` for slower. |
| `--pitch N` | Semitones, -12..12, on the finished sound: negative is lower and bigger (tone and "throat" move together). -3 to -5 is a good start. |
| `--tempo X` | Time-stretch, 0.5-1.5, same pitch: below 1 is slower. |
| `--reverb KIND` | `none` (dry: for a take that will get its own room in a mix), `room` (default), `hall`, `cathedral`. |
| `--house-voice [TAKE]` | Show the house voice, or set it to TAKE. |
| `--like TAKE` | Keep an earlier take's voice for a new line: use it for multi-line scenes so every line sounds like the same Reaper. Works across languages: Romanian lines can clone an English take's voice (and the reverse). |
| `--fx TAKE` | Re-run effects (preset, pitch, tempo, reverb) on an earlier take: no new generation, ~1-3 s. Use it to iterate on effects. |
| `--check` / `--no-check` | Whisper transcribes the dry and processed audio; prints words-heard scores and writes a spectrogram PNG into `output/`. **On by default**; `--no-check` saves ~1 s when you don't need it. |
| `--out DIR` / `--name STEM` | Copy the processed audio and the dry WAV to DIR (never overwrites). |
| `--format F` | With `--out`: `both` (default), `wav` or `mp3`. **WhatsApp and most messengers reject WAV**: give the user MP3 for those. Every take gets an MP3 automatically. |
| `--wait S` | Wait up to S seconds for the server. |
| `--json` | Only the server's JSON (plus `paths`): parse this rather than the text output. |

**Presets:**

| Preset | Use it for |
|---|---|
| `reaper-drone` | one unmoving brass pitch, the words shape a drone |
| `reaper` | the same machine, loosely following the voice's intonation |
| `harbinger` | the default: a synth chord snapping to semitones, ring-mod shimmer, heavy distortion |
| `demon` | organic and monstrous rather than mechanical: a demon more than a machine |
| `dry` | the base voice, no effects |

**Takes:** every generation is a *take* with an id like `20260101-120000-ab12`. Its files live in
`output/`: `<id>_dry.wav` (24 kHz, the voice as generated), `<id>_fx-<preset>[_pitch-4][_tempo0.85][_hall].wav`
and `.mp3` (48 kHz, processed; one pair per effects combination), `<id>.json` (text, instruction,
seed: everything to redo it), and a matching `.png` with `--check`. Responses may carry
`warnings` (the CLI prints them as `WARNING:`) and `check_error` (the take is fine, the check
failed): relay warnings to the user.

**Exit codes:** 0 done; 1 no server, lost the server, or a server error (the message says which);
2 bad arguments.

**Speed on the RTX 5080, warm:** English generation runs ~2-2.5x faster than real time (a 12 s
line in ~5 s); Romanian ~1-1.5 s a sentence including its check; effects ~0.2x real time;
Whisper ~1 s. **Length:** text is capped at 800 characters, and Breeze stops at 120 s of audio (a
take that hits it gets a warning: its end is missing). Keep a take to a few sentences and chain
lines with `--like`.

## Romanian

- **Romanian clones the house voice by default** (no `--instruction`, no `--like`): a deep
  English Breeze take (~65 Hz), made automatically on the first Romanian line (~30 s once) and
  stored in `output/house_voice.json`. Why: OmniVoice's own "very low pitch" lands around
  100-130 Hz, which sounds too small next to the English takes. `--house-voice`
  shows it; `--house-voice <take>` sets it (pick a deep English take). Only design Romanian
  voices from attributes if the user wants a different voice, and say it will sound higher.
- **Whisper can't hear an accent.** A cloned English voice might carry English colouring into
  Romanian; tell the user to listen for it, and offer attribute-designed Romanian as the alternative.
- **Sentence by sentence, verified:** the server generates Romanian one sentence at a time,
  transcribes each, and regenerates any sentence with missing words (up to 3 tries), because
  OmniVoice otherwise drops words, especially in long inputs. The response's `sentences` lists
  each sentence's score and tries; a warning means one still misses words after 3 tries: try
  another seed or rephrase that sentence.
- **Text:** write proper Romanian with diacritics (ă â î ș ț). Numbers, decimals ("3,5") and clock
  times ("10:30") are spelled out in Romanian automatically and cedilla letters (ş ţ) are fixed;
  abbreviations aren't expanded, so write them out. Stress errors don't show in transcripts either: if the user reports a wrong
  stress, rephrase or try another seed.
- The Romanian engine starts on the first Romanian request (~20 s, plus ~10 s once to make the
  house voice), then ~1-1.5 s a sentence.

## 3. Write the voice description

- **Describe a person's voice and delivery**, in English: age, size, timbre, pitch, pace, emotion,
  attitude. Breeze is trained on human speech, so "robotic" or "inhuman" mostly read as
  flat and cold. Leave the inhuman part to the presets.
- **The deeper and slower the base voice, the better the Reaper presets work.** Words like
  "massive scale", "subterranean", "deep resonant bass", "very slowly", "long pauses" pull the
  pitch down to ~65-70 Hz. A take whose base voice comes out higher (check the dry file) sounds
  less huge and is harder to understand after effects: just try another seed.
- **Mood goes in the description too:** "contemptuous", "patient, almost amused", "furious but
  controlled". The presets keep whatever performance the model gives.
- **Vocal events** work inline in the text: `(sigh)`, `(laugh)`, `(clears throat)`.

## 4. Judge every take

- **Words heard, dry:** below ~90% means the model fluffed the line itself (a skipped or
  mangled word). Change the seed.
- **Words heard, wet:** below ~85% with a good dry score means the effects are burying the speech.
  Try another seed first (it's often the base voice), then a lighter preset.
- **`[Whisper looped: unreliable, listen]`:** Whisper got stuck repeating a word. That says
  nothing about the audio either way. The spectrogram and the user's ears decide.
- **Whisper is a rough proxy near the threshold.** On heavily processed audio a tiny change can
  swing it from 0% to 90%. Don't tune presets to a single take's score: compare across 3-4 takes.
- **The spectrogram:** speech should show clear formant bands and gaps between words. A solid
  wall with no gaps means too much distortion or a vocoder that's smearing.

## 5. When you're done

The server keeps the model in VRAM: that's what makes the next take fast, but it blocks the
local-agent and local-image servers. **Leave it running and tell the user** it's up. Stop it when
they're done or want one of those:

```bash
cmd //c "<base dir>\\stop.cmd"
```

It stops the server recorded in `output/.server.pid` (and its Romanian worker), whichever path
started it, and nothing else. A second `start.cmd` while one runs exits at once with "port in
use": check `/api/health` first rather than starting twice. The page at http://127.0.0.1:8190/
does everything the CLI does.

## Tuning presets

Presets are JSON in `voicefx/presets/`: `pre` chain, parallel `buses` mixed at `gain_db`, `post`
chain. The vocoder's knobs (`f0`, `follow`, `steps`, `chord`, `metal`, `sub`) are documented in
`op_vocoder`; other ops are in `voicefx/fx.py` (`OPS`). Preset names must be letters, digits,
`-` or `_`. Name experiments `_something.json`: git ignores them. Edit or add one, then re-run
effects on existing takes with `--fx`: no restart needed, since presets are read on every request
(changes to `fx.py` itself need a server restart). Run `.venv\Scripts\python.exe -m pytest tests -q` after changing `fx.py`.
