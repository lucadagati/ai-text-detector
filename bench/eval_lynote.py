"""Benchmark the lightweight analyzer of github.com/lynote-ai/ai-text-detector on our set."""
import glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
EXT = os.environ.get("LYNOTE_DIR", "lynote-ai-text-detector/src")  # clone of github.com/lynote-ai/ai-text-detector
sys.path.insert(0, EXT)
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from aidetect import analyze_text  # noqa: E402  (lynote package)
from aidet.textprep import latex_to_text, split_paragraphs  # noqa: E402
D = os.path.join(os.path.dirname(__file__), "data")
norm = lambda t: " ".join(latex_to_text(t.replace("**", "")).split())
rows = []
for f, y in (("human.jsonl", 0), ("ai.jsonl", 1)):
    seen = set()
    for l in open(os.path.join(D, f)):
        r = json.loads(l)
        if r["id"] in seen: continue
        seen.add(r["id"])
        g = "human" if y == 0 else ("Claude" if r["gen"].startswith("claude") else "other LLMs") + (" polish" if r["mode"] == "polish" else "")
        rows.append((analyze_text(norm(r["text"])).score, y, g))
s = np.array([r[0] for r in rows], float); y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])
fp, tp, _ = roc_curve(y, s)
print(f"lynote lightweight analyzer: AUROC {roc_auc_score(y, s):.3f}  TPR@5%FPR {np.interp(.05, fp, tp):.3f}")
t5 = np.quantile(s[y == 0], .95)
for k in sorted(set(g)):
    print(f"  {k:20s} mean score {s[g == k].mean():5.1f}   flagged at 5%FPR {np.mean(s[g == k] > t5):.2f}")
for name, pat in [a.split("=", 1) for a in sys.argv[1:]]:
    v = [analyze_text(p).score for f in sorted(glob.glob(pat)) for p in split_paragraphs(open(f).read(), 40, latex=True)]
    print(f"  {name:20s} mean score {np.mean(v):5.1f}   flagged at 5%FPR {np.mean(np.array(v) > t5):.2f}  (n={len(v)})")
