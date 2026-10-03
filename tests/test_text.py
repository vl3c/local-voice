r"""Text handling: Romanian normalisation and the words-heard score.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from textnorm import normalise  # noqa: E402


@pytest.mark.parametrize("text, spoken", [
    ("Anul 2026, ţara.", "Anul două mii douăzeci și șase, țara."),
    ("Ora 10:30.", "Ora zece și treizeci."),
    ("Ora 10:00.", "Ora zece."),
    ("Ai 3,5 secunde.", "Ai trei virgulă cinci secunde."),
    ("1, 2, 3", "unu, doi, trei"),
])
def test_romanian_normalisation(text, spoken):
    pytest.importorskip("num2words")
    assert normalise(text, "ro") == spoken


def test_other_languages_untouched():
    assert normalise("In 2026 at 10:30", "en") == "In 2026 at 10:30"


@pytest.fixture(scope="module")
def app():
    import app as server  # noqa: PLC0415  (imports FastAPI and voicefx, loads no models)
    return server


def test_romanian_scoring_folds_cedillas(app):
    assert app.word_match("Știința țării", "Ştiinţa ţării") == 1.0


def test_vocal_events_are_ignored(app):
    assert app.word_match("(sigh) We are the dark.", "We are the dark") == 1.0


def test_hallucination_loop_detected(app):
    assert app.looks_hallucinated("Oh, oh, oh, oh, oh, oh!")
    assert not app.looks_hallucinated("You exist because we allow it. You will end because we demand it.")


def test_short_normal_sentences_are_not_loops(app):
    assert not app.looks_hallucinated("We are the end of everything.")
    assert not app.looks_hallucinated("Noi suntem începutul și sfârșitul.")
    assert app.looks_hallucinated("Oh shit, shit, shit. Oh shit, shit, shit.")
