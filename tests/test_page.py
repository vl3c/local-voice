r"""The page and the server agree: a static contract test of server/page.html (no browser).

It reads the page's script and checks that every element it looks up exists, that every
field it sends is one the server accepts, and that every file it links is one the server
returns. These are the mismatches that break a UI silently.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

PAGE = (ROOT / "server" / "page.html").read_text(encoding="utf-8")
SCRIPT = "\n".join(re.findall(r"<script>(.*?)</script>", PAGE, re.S))


def block(start: str) -> str:
    """The source from `start` to its matching closing brace."""
    i = SCRIPT.index(start)
    j = SCRIPT.index("{", i)
    depth = 0
    for k in range(j, len(SCRIPT)):
        depth += {"{": 1, "}": -1}.get(SCRIPT[k], 0)
        if depth == 0:
            return SCRIPT[i:k + 1]
    raise AssertionError(f"unbalanced block at {start!r}")


def code_only(src: str) -> str:
    """The script without comments and string contents (so prose like "it: ..." isn't a key)."""
    src = re.sub(r"//[^\n]*", "", src)
    return re.sub(r'"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`', '""', src)


def keys(src: str) -> set[str]:
    """Object keys written as `key:` plus assignments `body.key =` in a piece of script."""
    src = code_only(src)
    found = set(re.findall(r"[{,\s](\w+):\s", src)) | set(re.findall(r"body\.(\w+)\s*=", src))
    return found - {"method", "headers", "body", "class", "tip", "onclick", "href", "download", "src", "alt"}


@pytest.fixture(scope="module")
def app():
    import app as server  # noqa: PLC0415
    return server


def test_every_element_the_script_uses_exists():
    ids = set(re.findall(r'\bid="([\w-]+)"', PAGE))
    used = set(re.findall(r'\$\("([\w-]+)"\)', SCRIPT))
    assert used - ids == set()


def test_speak_sends_only_what_the_server_accepts(app):
    sent = keys(block('$("go").addEventListener')) | keys(block("function panelEffects"))
    assert sent - {"text"} <= set(app.SpeakReq.model_fields), sent - set(app.SpeakReq.model_fields)


def test_rerun_sends_only_what_fx_accepts(app):
    sent = keys(block("const rerunWith")) | keys(block("function panelEffects")) | {"trim_tail"}
    assert sent <= set(app.FxReq.model_fields), sent - set(app.FxReq.model_fields)


def test_saved_settings_are_settings_the_server_keeps(app):
    sent = keys(block("function collectSettings"))
    assert sent <= set(app.Settings.model_fields), sent - set(app.Settings.model_fields)
    assert set(app.Settings.model_fields) - sent == set(), "a setting the page can't save"


def test_the_romanian_voice_calls_use_real_kinds(app):
    for kind in re.findall(r'kind = [^;]*?"(\w+)" : "(\w+)"', SCRIPT)[0]:
        assert kind in app.RO_KINDS
    for value in re.findall(r'<option value="(\w+)">', PAGE[PAGE.index('id="roVoice"'):PAGE.index("</select>", PAGE.index('id="roVoice"'))]):
        assert value in ("auto", *app.RO_KINDS)


@pytest.fixture(scope="module")
def responses(tmp_path_factory):
    """What the server really returns: a checked English take, a Romanian take, an effects re-run
    (fake models, see test_server.py)."""
    import json as _json
    import test_server as ts
    import app as server
    tmp = tmp_path_factory.mktemp("page")
    mp = pytest.MonkeyPatch()
    try:
        for name, value in (("OUTPUT", tmp / "output"), ("VOICES", tmp / "voices"),
                            ("SETTINGS_FILE", tmp / "config.local.json")):
            mp.setattr(server, name, value)
        (tmp / "output").mkdir()
        breeze, whisper = ts.FakeBreeze(), ts.FakeWhisper()
        mp.setattr(server, "tts", lambda: breeze)
        mp.setattr(server, "asr", lambda: whisper)
        ts.fake_native(server, mp)
        server._fakes = (breeze, whisper)
        en = ts.speak(server, text="We are the dark.", check=True, blend=0.3, double_ms=10, bass_db=-3)
        ro = ts.speak(server, text="Bună seara.", language="ro", voice="young-woman")
        fx = ts.client(server).post("/api/fx", json={"id": en["id"], "check": False}).json()
        return [en, ro, {**_json.loads(server.take_paths(en["id"])["meta"].read_text(encoding="utf-8")), **fx}]
    finally:
        mp.undo()


def test_every_file_link_is_one_the_server_returns(responses):
    linked = set(re.findall(r"res\.files\.(\w+)", SCRIPT))
    returned = set().union(*(r["files"] for r in responses))
    assert linked <= returned, linked - returned


def test_every_field_shown_from_a_take_exists(responses):
    """The take card reads res.<field>: each must be something a real speak or fx returns."""
    shown = set(re.findall(r"res\.(\w+)", SCRIPT)) - {"files"}
    returned = set().union(*responses)
    assert shown <= returned, shown - returned


def test_every_reverb_and_option_is_valid(app):
    reverbs = set(re.findall(r'<option value="(\w+)"', PAGE[PAGE.index('id="reverb"'):PAGE.index("</select>", PAGE.index('id="reverb"'))]))
    import voicefx
    assert reverbs == set(voicefx.REVERBS)
