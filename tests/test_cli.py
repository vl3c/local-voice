r"""The CLI (scripts/say.py): options to requests, --out delivery, and the listing commands.
No server: the HTTP call is replaced by a fake.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import say  # noqa: E402


def request(*argv):
    return say.request_for(say.build_parser().parse_args(list(argv)))


def test_just_the_text_leaves_every_default_to_the_server():
    assert request("We are the dark.") == ("/api/speak", {"text": "We are the dark.", "language": "en"})


def test_options_map_to_the_api():
    path, body = request("Bună.", "--lang", "ro", "--voice", "vampire-male", "--ro-voice", "native", "--seed", "42",
                         "--preset", "dry", "--reverb", "none", "--blend", "0.3", "--bass", "-6", "--double", "20",
                         "--pitch", "-3", "--tempo", "0.9", "--speed", "0.8", "--no-check", "--trim-tail")
    assert path == "/api/speak" and body == {
        "text": "Bună.", "language": "ro", "voice": "vampire-male", "ro_voice": "native", "seed": 42,
        "preset": "dry", "reverb": "none", "blend": 0.3, "bass_db": -6.0, "double_ms": 20.0, "pitch": -3.0,
        "tempo": 0.9, "speed": 0.8, "check": False, "trim_tail": True}


def test_random_seed_is_null_and_no_trim_tail_is_false():
    _, body = request("x", "--seed", "random", "--no-trim-tail", "--instruction", "")
    assert body["seed"] is None and body["trim_tail"] is False and body["instruction"] == ""


def test_fx_sends_only_effects():
    path, body = request("--fx", "20260101-120000-ab12", "--preset", "wraith", "--voice", "child", "--lang", "ro")
    assert path == "/api/fx" and body == {"id": "20260101-120000-ab12", "preset": "wraith"}


def test_old_option_names_still_work():
    args = say.build_parser().parse_args(["--default-voice", "previous", "--format", "both"])
    assert args.ro_take == "previous" and args.format == "both"


def test_bad_arguments_exit_2(monkeypatch):
    for argv in (["x", "--seed", "abc"], [], ["x", "--ro-voice", "maybe"]):
        monkeypatch.setattr(sys, "argv", ["say.py", *argv])
        with pytest.raises(SystemExit) as e:
            say.main()
        assert e.value.code == 2


# ------------------------------------------------------------------ --out delivery

@pytest.fixture()
def take(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(say, "OUTPUT", out)
    files = {"dry": "t_dry.wav", "wet": "t_fx-x.wav", "mp3": "t_fx-x.mp3", "voice": "t_fx-x_voice.wav",
             "words": "t_fx-x_words.json", "srt": "t_fx-x.srt", "spectrogram": "t_fx-x.png"}
    for name in files.values():
        (out / name).write_text(name)
    return {"files": files}, tmp_path / "deliver"


def test_all_delivers_everything_but_the_spectrogram(take):
    res, dest = take
    names = sorted(p.name for p in say.deliver(res, dest, "L17", "all") if p.parent == dest)
    assert names == ["L17.mp3", "L17.srt", "L17.wav", "L17_dry.wav", "L17_voice.wav", "L17_words.json"]


def test_formats_pick_the_files(take):
    res, dest = take
    assert [p.name for p in say.deliver(res, dest / "m", "a", "mp3") if p.parent == dest / "m"] == ["a.mp3"]
    wav = sorted(p.name for p in say.deliver(res, dest / "w", "a", "wav") if p.parent == dest / "w")
    assert "a.mp3" not in wav and "a.wav" in wav and "a_words.json" in wav


def test_delivery_never_overwrites(take):
    res, dest = take
    say.deliver(res, dest, "L1", "mp3")
    again = [p.name for p in say.deliver(res, dest, "L1", "mp3") if p.parent == dest]
    assert again == ["L1_1.mp3"]


def test_without_out_the_paths_point_into_output(take):
    res, _ = take
    assert all(p.parent == say.OUTPUT for p in say.deliver(res, None, None))


# ------------------------------------------------------------------ main() against a fake server

@pytest.fixture()
def fake_server(monkeypatch):
    calls = []
    replies = {}

    def call(path, payload=None, timeout=900, method=None):
        calls.append((path, payload, method))
        for prefix, reply in replies.items():
            if path.startswith(prefix):
                return reply(path, payload) if callable(reply) else reply
        raise AssertionError(f"unexpected call {path}")

    monkeypatch.setattr(say, "call", call)
    monkeypatch.setattr(say, "wait_for_server", lambda seconds: True)
    return calls, replies


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["say.py", *argv])
    status = say.main()
    return status, capsys.readouterr().out


def test_voices_lists_both_romanian_kinds(fake_server, monkeypatch, capsys):
    _, replies = fake_server
    replies["/api/voices"] = {"young-woman": {"name": "Young woman", "seed": 1, "about": "Calm.", "instruction": "A woman.",
                                              "romanian": {"use": "native"},
                                              "romanian_takes": {"clone": None, "native": {"id": "T1"}}}}
    status, out = run(monkeypatch, capsys, "--voices")
    assert status == 0 and "clone: made on first use" in out and "native (auto): take T1" in out


def test_ro_take_shows_sets_and_undoes_per_kind(fake_server, monkeypatch, capsys):
    calls, replies = fake_server
    replies["/api/default-voice"] = {"id": "T9", "f0_hz": 180.0, "previous": {"id": "T8", "f0_hz": 170.0}}
    run(monkeypatch, capsys, "--ro-take", "--voice", "child", "--ro-voice", "native")
    assert calls[-1][0] == "/api/default-voice?voice=child&kind=native"
    run(monkeypatch, capsys, "--ro-take", "T9", "--voice", "child")
    assert calls[-1][:2] == ("/api/default-voice", {"id": "T9", "voice": "child", "kind": "auto"})
    run(monkeypatch, capsys, "--default-voice", "T9", "--voice", "young-woman")  # the old name, the old usage
    assert calls[-1][:2] == ("/api/default-voice", {"id": "T9", "voice": "young-woman", "kind": "auto"})
    status, out = run(monkeypatch, capsys, "--ro-take", "previous")
    assert calls[-1][:2] == ("/api/default-voice/restore-previous", {"voice": "ancient-male", "kind": "auto"})
    assert "take T9" in out and "--ro-take previous" in out


def test_a_take_prints_its_summary_and_warnings(fake_server, monkeypatch, capsys, tmp_path):
    _, replies = fake_server
    monkeypatch.setattr(say, "OUTPUT", tmp_path)
    replies["/api/speak"] = {"id": "T1", "seed": 7, "preset": "dry", "reverb": "none", "language": "ro",
                             "voice": "vampire-male", "default_voice": True, "ro_voice": "native", "like": "T0",
                             "files": {"wet": "T1.wav"}, "duration_s": 2.0, "gen_s": 1.0, "fx_s": 0.2,
                             "heard": {"dry": "Bună", "wet": "Bună"}, "match": {"dry": 1.0, "wet": 1.0},
                             "words": [{"word": "Bună", "start": 0.1, "end": 0.5, "approx": True}],
                             "warnings": ["the words sit low under the bass"]}
    status, out = run(monkeypatch, capsys, "Bună.", "--lang", "ro", "--voice", "vampire-male")
    assert status == 0
    assert "voice: vampire-male, native Romanian (take T0)" in out
    assert "word timings: 1 words (1 approximate)" in out and "WARNING: the words sit low" in out


def test_json_output_carries_the_delivered_paths(fake_server, monkeypatch, capsys, tmp_path):
    _, replies = fake_server
    monkeypatch.setattr(say, "OUTPUT", tmp_path)
    (tmp_path / "T1.mp3").write_text("x")
    replies["/api/speak"] = {"id": "T1", "files": {"mp3": "T1.mp3"}}
    status, out = run(monkeypatch, capsys, "x", "--json", "--out", str(tmp_path / "d"), "--format", "mp3",
                      "--name", "hello")
    data = json.loads(out)
    assert status == 0 and data["paths"] == [str(tmp_path / "d" / "hello.mp3")]


def test_no_save_deletes_the_take_or_only_the_preview(fake_server, monkeypatch, capsys, tmp_path):
    calls, replies = fake_server
    monkeypatch.setattr(say, "OUTPUT", tmp_path)
    replies["/api/speak"] = {"id": "T1", "files": {"wet": "T1.wav"}}
    replies["/api/fx"] = {"id": "T1", "files": {"dry": "T1_dry.wav", "wet": "T1_fx.wav", "mp3": "T1_fx.mp3"}}
    replies["/api/takes/"] = {"id": "T1", "deleted": 5}
    replies["/files/"] = {"deleted": 1}
    run(monkeypatch, capsys, "x", "--no-save")
    assert calls[-1] == ("/api/takes/T1", None, "DELETE")
    run(monkeypatch, capsys, "--fx", "T1", "--no-save")
    deleted = [c[0] for c in calls if c[2] == "DELETE" and c[0].startswith("/files/")]
    assert deleted == ["/files/T1_fx.wav", "/files/T1_fx.mp3"]  # never the take's own dry voice
