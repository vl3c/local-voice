# Model licences

This repo's own code is MIT (see `LICENSE`). It does **not** include or
redistribute any model weights or third-party code: `install.ps1` fetches them
from their upstream sources, and they stay under their own licences.

| Component | Source | Licence | What it means here |
|---|---|---|---|
| Breeze TTS 2 weights | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | BreezeBlue Research and Non-Commercial License (copy in `models/breeze-tts-2/LICENSE` after install) | Research and non-commercial use only, **including the audio you generate with it** ("self-hosted outputs"). Commercial use needs written permission from BreezeBlue. |
| Breeze TTS 2 inference code | [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) (cloned into `engine/`) | Apache 2.0 | Its audio tokenizer is based on Qwen3-TTS (Apache 2.0). |
| OmniVoice weights | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | CC-BY-NC (version not stated; "due to constraints from its training data") | The Romanian (and other-language) engine. Non-commercial; credit k2-fsa if you share outputs. |
| OmniVoice code | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (installed into `.venv-ro`) | Apache 2.0 | |
| num2words | [savoirfairelinux/num2words](https://github.com/savoirfairelinux/num2words) | LGPL-2.1 | Spells out numbers in Romanian. Installed as a library, not redistributed. |
| Whisper large-v3-turbo | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | MIT | Used only to transcribe takes for checking. |
| Pedalboard | [spotify/pedalboard](https://github.com/spotify/pedalboard) | GPL-3.0 | Installed into the venv as a library; not redistributed. |
| Parselmouth / Praat | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) | GPL-3.0 | Same. |

This repo distributes only its own MIT code: install.ps1 fetches every library and model from its
upstream. If you redistribute a bundle that includes Pedalboard or Parselmouth (a packaged build,
an installer with the venv inside), the GPL-3.0 applies to that bundle.

The effects presets in `voicefx/presets/` are original. Their starting points came from
published sound-design research, used as ideas only (no audio or data from either is included):
the [Designed Vocalizations Dataset](https://arxiv.org/abs/2607.20951) (NC AI / NCSOFT,
Interspeech 2026; [project page](https://ncai-official.github.io/speech/publications/designed-vocalizations-dataset/))
and its preset chains, and [VoiceDesigner](https://arxiv.org/abs/2608.13613)'s character
augmentation chains. The `ai-choir` creature's spectrum and layering were measured, by ear and
by analysis, from a short game clip kept locally and never included.

Mass Effect and the Reapers belong to Electronic Arts / BioWare. The presets
imitate a style of sound processing; they contain no audio from the games.
