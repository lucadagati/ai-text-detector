"""Tests that need neither a GPU nor the model weights."""
import os, subprocess, sys, types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from aidet.textprep import accept_changes, normalize_unicode, split_paragraphs  # noqa: E402
import humanize  # noqa: E402
import score_tex  # noqa: E402

LONG = " ".join(["word"] * 60)


def test_normalize_unicode_removes_invisible_and_homoglyphs():
    assert normalize_unicode("pаper​") == "paper"  # Cyrillic a, zero-width space


def test_accept_changes():
    t = r"keep \added{new} \deleted{old} \replaced{fresh}{stale} end"
    assert accept_changes(t).split() == ["keep", "new", "fresh", "end"]


def test_split_paragraphs_drops_short_and_math_heavy():
    text = f"{LONG}\n\nshort one\n\n{LONG}"
    assert len(split_paragraphs(text, 40, False)) == 2


def test_blocks_keep_environments_whole():
    src = "a b\n\n\\begin{table}\nx\n\ny\n\\end{table}\n\nlast"
    assert [b[1].count("\n") for b in score_tex.blocks(src)] == [0, 3, 0]


def test_invariants_detect_changed_refs_numbers_math():
    base = r"We run 3 trials \cite{a} with $x_1$."
    assert humanize.invariants(base) == humanize.invariants(r"With $x_1$, we run 3 trials \cite{a}.")
    assert humanize.invariants(base) != humanize.invariants(r"We run 4 trials \cite{a} with $x_1$.")
    assert humanize.invariants(base) != humanize.invariants(r"We run 3 trials \cite{b} with $x_1$.")
    assert humanize.invariants(base) != humanize.invariants(r"We run 3 trials \cite{a} with $x_2$.")


def test_humanizer_rules_loaded_without_front_matter():
    r = humanize.rules()
    assert not r.startswith("---") and "Humanizer" in r


def test_track_changes_roundtrip(tmp_path):
    old = tmp_path / "old.tex"
    new = tmp_path / "new.tex"
    out = tmp_path / "out.tex"
    old.write_text("The system is fast. It uses a cache.\n\nSecond paragraph stays.\n")
    new.write_text("The system is quick. It uses a cache.\n\nSecond paragraph stays.\n\nA new paragraph.\n")
    subprocess.run([sys.executable, os.path.join(ROOT, "track_changes.py"), str(old), str(new), str(out)], check=True)
    accepted = " ".join(accept_changes(out.read_text()).split())
    assert accepted == " ".join(new.read_text().split())


def test_server_limits(monkeypatch):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import server
    monkeypatch.setattr(server, "MAX_CHARS", 100)
    c = TestClient(server.app)
    assert c.post("/api/detect", json={"text": "x" * 101}).status_code == 413
    assert c.post("/api/detect", json={"text": "x", "min_words": 0}).status_code == 422
    assert c.get("/api/health").json()["status"] == "ok"


def test_document_verdict_rules(monkeypatch):
    from aidet import ensemble
    from aidet.ensemble import Detector
    monkeypatch.setattr(ensemble, "structure_distance", lambda paras: None)
    monkeypatch.setitem(sys.modules, "aidet.syntax", types.SimpleNamespace(profile=lambda p: ({}, 0)))  # no spaCy in CI

    def para(label):
        return {"text": LONG, "words": 60, "score": 1.5 if label == "ai" else -1, "label": label, "signals": {}, "sentences": []}

    det = Detector()
    verdict = lambda labels: det._summary([], 0, [para(x) for x in labels])["verdict"]  # noqa: E731
    assert verdict(["human"] * 10) == "likely human"
    assert verdict(["ai"] + ["human"] * 9).startswith("likely human (isolated")
    assert verdict(["ai"] * 2 + ["human"] * 8).startswith("likely human (isolated")
    assert verdict(["ai"] * 3 + ["human"] * 17).startswith("mixed")
    assert verdict(["ai"] * 3 + ["human"] * 7) == "likely AI-generated"
    assert verdict([]) == "too short"
