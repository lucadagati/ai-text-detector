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


def test_humanizer_rules_compact_by_default_full_on_request(monkeypatch):
    compact = humanize.rules()
    assert len(compact) < 3000 and not compact.startswith("---")
    monkeypatch.setattr(humanize, "FULL_RULES", True)
    full = humanize.rules()
    assert not full.startswith("---") and "Humanizer" in full and len(full) > 10 * len(compact)


def test_prompt_carries_academic_guidelines_and_measured_gaps():
    d = {"paragraphs": [{"sentences": [{"text": "x", "score": 3.0}]}], "sentence_threshold_5": 0.0,
         "structure_traits": [{"trait": "passive %", "value": 60.0, "human_mean": 34.0, "z": 1.9}]}
    p = humanize.build_prompt("A paragraph.", d, None, True)
    assert "ACADEMIC WRITING GUIDELINES" in p and "passive %: 60.0 vs human papers 34.0" in p and "active clauses" in p


def test_directives_follow_direction_and_threshold():
    mk = lambda t, v, h, z: {"structure_traits": [{"trait": t, "value": v, "human_mean": h, "z": z}]}  # noqa: E731
    assert "active clauses" in humanize.directives(mk("passive %", 60, 34, 2.0))
    assert "passive where the agent is irrelevant" in humanize.directives(mk("passive %", 5, 34, -2.0))
    assert humanize.directives(mk("passive %", 40, 34, 0.8)) == ""          # below |z| 1.5
    assert humanize.directives(mk("unknown trait", 1, 0, 3.0)) == ""
    assert humanize.directives(mk("ends with ', not X' %", 1, 5, -2.0)) == ""  # no instruction for the low side


def test_math_share_and_tier_budget():
    assert humanize.math_share("no math here") == 0
    assert humanize.math_share("$abcdef$ xx") > 0.5
    humanize.CALLS.clear()
    humanize.BUDGET.clear()
    humanize.BUDGET["opus"] = 1
    tiers = ["sonnet", "opus"]
    assert humanize.pick_tier(tiers, 0, False) == "sonnet"          # cheap tier first
    assert humanize.pick_tier(tiers, 1, True) == "opus"             # moves up after a failed round
    assert humanize.pick_tier(tiers, 1, True) == "sonnet"           # opus budget spent: falls back
    humanize.CALLS.clear()
    humanize.BUDGET.clear()


def test_academic_lint_rejects_formulas_hype_contractions_and_term_swaps():
    import academic_lint as L
    old = r"The \texttt{gate_check} runs at RTT scale. We measure 3 runs."
    assert L.violations(old, old) == []
    assert "added stock formulas" in L.violations(old, old + " It is important to note that this plays a crucial role.")
    assert "added hype words" in L.violations(old, old + " The result is remarkable.")
    assert "added contractions" in L.violations(old, old + " It doesn't fail.")
    swapped = L.violations(old, "The check runs at latency scale. We measure 3 runs.")
    assert swapped and swapped[0].startswith("terminology changed") and "RTT" in swapped[0]
    longer = old.replace("We measure 3 runs.", "We measure 3 runs, which is a thing that we did in order to know it better than before.")
    assert "sentences got longer" in L.violations(old, longer)


def test_academic_lint_ignores_cite_and_math_in_sentence_stats():
    import academic_lint as L
    r = L.lint(r"We prove $x_1 \le y$ in \cite{a}. It holds. It stays.")
    assert r["formulas"] == 0 and r["mean_len"] > 0


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


def test_cli_fail_above_gate(monkeypatch, capsys):
    import cli

    class Resp:
        def read(self):
            import json
            p = {"words": 50, "score": 2.0, "label": "ai", "text": "x", "sentences": []}
            return json.dumps({"verdict": "likely AI-generated", "share_words": {"ai": 1.0, "uncertain": 0.0, "human": 0.0},
                               "n_paragraphs": 1, "paragraphs": [p], "sentence_threshold_5": 0.0}).encode()

    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *a, **k: Resp())
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(read=lambda: "text"))
    for extra in ([], ["--json"]):
        monkeypatch.setattr(sys, "argv", ["aidetect", "-", "--fail-above", "20", *extra])
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.value.code == 1
    monkeypatch.setattr(sys, "argv", ["aidetect", "-", "--fail-above", "100"])
    cli.main()
