"""Text clean-up before TTS. Imported by omni_worker.py; no heavy dependencies."""

from __future__ import annotations

import re

# Romanian: comma-below letters are correct; the legacy cedilla forms (ş ţ) still
# turn up in pasted text and quietly degrade TTS pronunciation.
_CEDILLA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})


def normalise(text: str, language: str) -> str:
    """Spell out numbers (incl. decimals and clock times) in Romanian; fix cedilla letters."""
    if language != "ro":
        return text
    text = text.translate(_CEDILLA)
    try:
        from num2words import num2words
    except ImportError:
        return text

    def n(s: str) -> str:
        return num2words(int(s), lang="ro")

    # 10:30 -> "zece și treizeci"; 10:05 -> "zece și cinci"; 10:00 -> "zece"
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: n(m[1]) + ("" if int(m[2]) == 0 else " și " + n(m[2])), text)
    # 3,5 -> "trei virgulă cinci" (Romanian decimal comma; "3, 5" in a list is left alone)
    text = re.sub(r"\b(\d+),(\d+)\b", lambda m: f"{n(m[1])} virgulă {n(m[2])}", text)
    return re.sub(r"\d+", lambda m: n(m[0]), text)
