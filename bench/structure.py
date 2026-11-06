"""Sentence-structure comparison: human arXiv vs LLM paragraphs, plus given manuscripts.

  structure.py [--doc NAME=GLOB ...]
Prints the mean of every structural feature per group, how well each feature alone separates
human from Claude paragraphs (AUROC), and a structure-only classifier (grouped 5-fold CV)
that is then applied to the paragraphs of each --doc.
"""
import argparse, glob, json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.syntax import paragraph_features  # noqa: E402
from aidet.textprep import latex_to_text, split_paragraphs  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(D, "syntax.jsonl")


def norm(t):
    return " ".join(latex_to_text(t.replace("**", "")).split())


def cached(items):
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l)
            cache[r["id"]] = r["f"]
    new = 0
    with open(CACHE, "a") as fh:
        for it in items:
            if it["id"] not in cache:
                cache[it["id"]] = paragraph_features(it["text"])
                fh.write(json.dumps({"id": it["id"], "f": cache[it["id"]]}) + "\n")
                new += 1
    for it in items:
        it["f"] = cache[it["id"]]
    return [it for it in items if it["f"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[], help="NAME=GLOB of .tex files")
    a = ap.parse_args()
    items = []
    for l in (l for n in ("human.jsonl", "human_abs.jsonl") if os.path.exists(os.path.join(D, n)) for l in open(os.path.join(D, n))):
        r = json.loads(l)
        items.append({"id": r["id"], "paper": r["paper"], "group": "human arXiv", "y": 0, "text": norm(r["text"])})
    for l in open(os.path.join(D, "ai.jsonl")):
        r = json.loads(l)
        g = ("Claude" if r["gen"].startswith("claude") else "other LLMs") + (" (polish)" if r["mode"] == "polish" else "")
        items.append({"id": r["id"], "paper": r["paper"], "group": g, "y": 1, "text": norm(r["text"])})
    docs = {}
    for spec in a.doc:
        name, pat = spec.split("=", 1)
        for f in sorted(glob.glob(pat)):
            for i, p in enumerate(split_paragraphs(open(f).read(), 40, latex=True)):
                items.append({"id": f"{name}:{os.path.basename(f)}#{i}", "paper": name, "group": name, "y": None, "text": p})
        docs[name] = True
    items = cached({it["id"]: it for it in items}.values())
    keys = list(items[0]["f"].keys())
    groups = ["human arXiv", "Claude", "Claude (polish)", "other LLMs"] + list(docs)
    F = {g: np.array([[it["f"][k] for k in keys] for it in items if it["group"] == g]) for g in groups}

    hv = F["human arXiv"]; cv = F["Claude"]
    print(f"{'paragraph-level mean':58s}" + "".join(f"{g[:14]:>15s}" for g in groups) + f"{'AUROC h/Claude':>16s}")
    print(f"{'(paragraphs)':58s}" + "".join(f"{len(F[g]):15d}" for g in groups))
    for j, k in enumerate(keys):
        y = np.r_[np.zeros(len(hv)), np.ones(len(cv))]
        auc = roc_auc_score(y, np.r_[hv[:, j], cv[:, j]])
        print(f"{k[:58]:58s}" + "".join(f"{F[g][:, j].mean():15.2f}" for g in groups) + f"{max(auc, 1 - auc):16.2f}")

    lab = [it for it in items if it["y"] is not None]
    X = np.array([[it["f"][k] for k in keys] for it in lab]); y = np.array([it["y"] for it in lab])
    grp = np.array([it["paper"] for it in lab]); src = np.array([it["group"] for it in lab])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        oof[te] = mk().fit(X[tr], y[tr]).decision_function(X[te])
    t5 = np.quantile(oof[y == 0], 0.95)
    f, t, _ = roc_curve(y, oof)
    print(f"\nstructure-only classifier (5-fold, grouped by paper): AUROC {roc_auc_score(y, oof):.3f}, "
          f"TPR@5%FPR {np.interp(.05, f, t):.3f}")
    for g in ["Claude", "Claude (polish)", "other LLMs"]:
        m = (src == "human arXiv") | (src == g)
        print(f"  human vs {g:16s} AUROC {roc_auc_score(y[m], oof[m]):.3f}   flagged at 5%FPR {np.mean(oof[src == g] >= t5):.3f}")
    clf = mk().fit(X, y)
    coef = clf[-1].coef_[0]
    print("\nmost AI-indicative structure (standardized weights):")
    for j in np.argsort(-np.abs(coef))[:10]:
        print(f"  {'+' if coef[j] > 0 else '-'}{abs(coef[j]):.2f}  {keys[j]}")
    thr = np.quantile(clf.decision_function(X[y == 0]), 0.95)
    for d in docs:
        Xd = F[d]
        s = clf.decision_function(Xd)
        print(f"\n{d}: {np.mean(s >= thr):.0%} of {len(s)} paragraphs flagged as AI by structure (threshold: 5% of human paragraphs)")


if __name__ == "__main__":
    main()
