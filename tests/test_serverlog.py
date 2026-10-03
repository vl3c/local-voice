r"""The server's console filter and log.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

import serverlog  # noqa: E402


def tee():
    console, log = io.StringIO(), io.StringIO()
    return serverlog.Tee(console, log, threading.Lock()), console, log


def test_noise_goes_to_the_log_only():
    t, console, log = tee()
    t.write("Warming up backbone graph (3 runs)...\n")
    t.write("Backbone CUDA graph captured!\n")
    t.write("W1004 01:28:14 x.py:6833] _maybe_guard_rel() was called on non-relation expression\n")
    t.write("[local-voice] ready on http://127.0.0.1:8190/\n")
    assert console.getvalue() == "[local-voice] ready on http://127.0.0.1:8190/\n"
    assert "Warming up backbone graph" in log.getvalue() and "_maybe_guard_rel" in log.getvalue()


def test_errors_and_unknown_output_reach_the_window():
    t, console, _ = tee()
    t.write("Traceback (most recent call last):\n  File \"app.py\", line 1\nRuntimeError: boom\n")
    t.write("something the filter has never seen\n")
    shown = console.getvalue()
    assert "Traceback" in shown and "RuntimeError: boom" in shown and "never seen" in shown


def test_progress_bar_redraws_collapse_to_the_final_state():
    t, console, log = tee()
    for pct in (0, 50, 100):
        t.write(f"\rLoading weights: {pct}%")
    t.write("\n")
    assert log.getvalue().count("Loading weights") == 1 and "100%" in log.getvalue()
    assert console.getvalue() == ""  # a known noisy bar: log only


def test_partial_writes_are_joined_into_lines():
    t, console, _ = tee()
    t.write("[local-voice] en take ")
    t.write("20260101-120000-aaaa: 5.0s of audio\n")
    assert console.getvalue() == "[local-voice] en take 20260101-120000-aaaa: 5.0s of audio\n"


def test_install_keeps_the_previous_log(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    (tmp_path / "server.log").write_text("old run\n", encoding="utf-8")
    path = serverlog.install(tmp_path)
    print("[local-voice] hello")
    sys.stdout.flush()
    assert (tmp_path / "server.previous.log").read_text(encoding="utf-8") == "old run\n"
    assert "[local-voice] hello" in path.read_text(encoding="utf-8")
