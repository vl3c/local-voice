r"""The repo scripts: the pre-publish safety scan and the verifying model downloader.

    .venv\Scripts\python.exe -m pytest tests -q
"""

import hashlib
import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ check_repo_safe

@pytest.fixture(scope="module")
def safe():
    return load("check_repo_safe")


def findings(safe, text):
    out = []
    safe.scan_text(text, "f", out)
    return [p.split(": ", 2)[1] for p in out]


# The samples are assembled at runtime, so this file itself never contains what the scan rejects.
@pytest.mark.parametrize("text, label", [
    ("path = C:" + "\\Users\\alice\\models", "Windows user path"),
    ("see /home" + "/alice/x", "Unix home path"),
    ("mail alice.smith" + "@corp.example.org", "email address"),
    ("host 192.168" + ".1.20", "IPv4 address"),
    ("key AKIA" + "ABCDEFGHIJKLMNOP", "AWS key"),
    ("-----BEGIN RSA " + "PRIVATE KEY-----", "private key block"),
    ("token ghp_" + "a" * 36, "GitHub token"),
    ("hf_" + "b" * 34, "HuggingFace token"),
    ("API_KEY" + "=abcd1234efgh", "secret assignment"),
    ("password" + ' = "hunter22hunter"', "secret assignment"),
    ("on DESKTOP-" + "AB12CD3 today", "machine name"),
])
def test_the_scan_catches(safe, text, label):
    assert label in findings(safe, text)


@pytest.mark.parametrize("text", [
    "C:\\Users\\<you>\\x", "you@example.com", "Co-Authored-By: x <noreply@anthropic.com>",
    "1234+someone@users.noreply.github.com", "http://127.0.0.1:8190/", "bind 0.0.0.0",
    'TOKEN = re.compile(r"[a-z]+")', "secret = get_secret()", "the password is not stored",
])
def test_the_scan_allows(safe, text):
    assert findings(safe, text) == []


def test_the_scan_runs_clean_on_this_repo(safe, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_repo_safe.py"])
    assert safe.main() == 0, capsys.readouterr().out


# ------------------------------------------------------------------ download_models

class FakeResponse(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}


@pytest.fixture()
def dl(tmp_path, monkeypatch):
    mod = load("download_models")
    data = {"a.bin": b"weights-a" * 100, "b.json": b'{"b": 1}'}
    spec = {"repos": [{"name": "m", "source": "https://example.invalid/m", "revision": "0" * 40,
                       "base_url": "https://example.invalid/m/resolve/x", "dest": "models/m",
                       "files": [{"path": k, "bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                                 for k, v in data.items()]}]}
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "models.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(mod, "REPO", repo)
    served = []

    def urlopen(req, timeout=60):
        name = req.full_url.rsplit("/", 1)[-1]
        served.append((name, req.get_header("Range")))
        return FakeResponse(data[name])

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    mod.data, mod.served, mod.root = data, served, tmp_path / "root"
    return mod


def run(dl, monkeypatch, *argv):
    accept = [] if "--verify" in argv or "--no-accept" in argv else ["--accept-licenses"]
    argv = [a for a in argv if a != "--no-accept"]
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--root", str(dl.root), *accept, *argv])
    return dl.main()


def test_nothing_is_fetched_until_the_licences_are_accepted(dl, monkeypatch, capsys):
    assert run(dl, monkeypatch, "--no-accept") == 1 and dl.served == []
    assert "--accept-licenses" in capsys.readouterr().out


def test_downloads_and_verifies(dl, monkeypatch):
    assert run(dl, monkeypatch) == 0
    assert (dl.root / "models/m/a.bin").read_bytes() == dl.data["a.bin"]
    assert run(dl, monkeypatch) == 0 and len(dl.served) == 2  # the second run finds both present


def test_verify_reports_missing_and_corrupt_files(dl, monkeypatch):
    assert run(dl, monkeypatch, "--verify") == 1 and dl.served == []
    run(dl, monkeypatch)
    (dl.root / "models/m/b.json").write_bytes(b'{"b": 2}')  # same size, different bytes
    assert run(dl, monkeypatch, "--verify") == 1


def test_a_bad_checksum_is_discarded(dl, monkeypatch):
    dl.data["a.bin"] = b"tampered!" * 100
    assert run(dl, monkeypatch, "--only", "m") == 1
    assert not (dl.root / "models/m/a.bin").exists()


def test_reuse_hard_links_matching_files(dl, monkeypatch, tmp_path):
    other = tmp_path / "other" / "models" / "m"
    other.mkdir(parents=True)
    for k, v in dl.data.items():
        (other / k).write_bytes(v)
    assert run(dl, monkeypatch, "--reuse", str(tmp_path / "other" / "models")) == 0
    assert dl.served == []  # nothing downloaded


def test_a_network_error_keeps_going_and_fails(dl, monkeypatch):
    def broken(req, timeout=60):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(dl.urllib.request, "urlopen", broken)
    assert run(dl, monkeypatch) == 1


def test_human_sizes():
    mod = load("download_models")
    assert mod.human(512) == "512.0 B" and mod.human(3 * 1024 ** 3) == "3.0 GB"
