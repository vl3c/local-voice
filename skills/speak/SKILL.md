---
name: speak
description: Talk to the user out loud through their speakers, in their local machine voice (English or Romanian), generated on this machine's GPU by the local-voice TTS. Use when the user asks you to speak, talk to them, say or read something out loud, answer by voice, or announce when a task is done; and keep using it for replies in a conversation where they asked for spoken answers, until they say to stop.
---

# speak: say things out loud to the user

This skill plays speech on the user's own speakers. It uses the **local-voice** tool, installed as
the `local-voice` skill at `~/.claude/skills/local-voice` (its `SKILL.md` is the full reference;
you don't need it for speaking). Use the commands below exactly as written: `~` works in Bash, and
`cygpath -w` turns the same path into the Windows path `cmd` needs. Don't build the path from this
skill's own folder with `..`: the skills are links, and `..` resolves to the wrong place.

## When to speak

- **Only when the user asked for it**: "say it out loud", "talk to me", "tell me when you're done",
  "read this to me", "answer by voice". Once they've asked for spoken replies, keep speaking your
  short replies until they say stop (or "quiet", "text only"). Never start speaking unprompted.
- **Always write the same thing in chat too.** Speech is in addition to text, never instead of it:
  the user may have stepped away, or misheard the machine voice.
- **Speak a summary, not the whole answer.** One to three sentences: the news, the result, or the
  question you need answered. Code, lists, tables, paths and long explanations stay in the chat.
- **Think about who else hears it.** The audio plays out loud in the room. Don't speak passwords,
  keys, private messages or other sensitive details, even if they're in your answer.

## How to speak

1. **Is the voice server up?**

   ```bash
   curl -s http://127.0.0.1:8190/api/health
   ```

   JSON means yes: go to step 3.

2. **If not, start it.** It needs most of the 16 GB GPU, so first check nothing else holds it:

   ```bash
   tasklist | grep -i -E "llama-server|python"
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
   ```

   If the local-agent server (`llama-server.exe`) or local-image's ComfyUI is running, **ask the
   user** before stopping it (with that skill's own `stop.cmd`); they may be using it. Then start
   the voice server. It opens in its own window (titled "local-voice server - close this window to
   stop it") and the command returns at once:

   ```bash
   cmd //c "$(cygpath -w ~/.claude/skills/local-voice/start.cmd)" --window
   ```

   In PowerShell instead: `& "$env:USERPROFILE\.claude\skills\local-voice\start.cmd" --window`.
   Tell the user in chat that the voice is warming up (about a minute the first time) and that
   closing that window stops it. Step 3's `--wait 180` waits for it to be ready.
   If `~/.claude/skills/local-voice` doesn't exist, or its `.venv` or `models` folder is missing,
   the tool isn't installed: tell the user, and don't run its ~23 GB installer without their
   go-ahead.

3. **Say it:**

   ```bash
   python ~/.claude/skills/local-voice/scripts/say.py "<what to say>" --play --no-check --no-save --wait 180
   ```

   Romanian: add `--lang ro`. `--no-save` deletes the audio once it has played, so speaking
   doesn't fill the disk (if the user asks to keep something you said, drop `--no-save` and give
   them the file). The command returns when the audio has finished playing. It prints
   `played <file>` on success. Exit code 1 means it didn't play (no server, or no audio player):
   tell the user in text that the voice didn't work, and why.

That's all. Every other setting (voice, preset, strength, seed, reverb) comes from the user's saved
defaults or the built-in ones. Don't override them unless the user asks: "say it in the harbinger
voice" means `--preset harbinger`; "lower" means `--pitch -3`; "slower" means `--tempo 0.85`.
`python ~/.claude/skills/local-voice/scripts/say.py --settings` shows what's in effect.

## Writing for the ear

- **Plain sentences only.** No markdown, bullet points, emoji, code, URLs or file paths: they get
  read out literally or mangled. Say "the config file" rather than its file name.
- **Short sentences, ordinary words.** The default voice is slow and heavy: about a second per
  word. Two sentences is often 10-15 seconds of speech. For longer passages, speak them as several
  calls of a sentence or two each (one call can't exceed 800 characters or about 2 minutes).
- **Spell out what a reader would skip over:** "version two point three" rather than "v2.3",
  "about forty percent" rather than "~40%". Romanian: use the diacritics (ă â î ș ț); numbers
  are spelled out for you.
- **Match the language** to the conversation: `--lang ro` when you're talking Romanian with the
  user.

## When you're done

Leave the server running: it keeps the voice ready, so the next line takes seconds. It does hold
the GPU (10-14 GB), which blocks the local-agent and local-image servers. If the user switches to
one of those, offer to stop the voice:

```bash
cmd //c "$(cygpath -w ~/.claude/skills/local-voice/stop.cmd)"
```
