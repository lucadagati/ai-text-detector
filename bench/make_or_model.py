"""Write aidet/model.json for the OR ensemble: a paragraph is `ai` if ANY member exceeds its own threshold at
alpha/k false positives (union bound: overall FPR <= alpha), `uncertain` likewise at 5%/k.

Thresholds come from the calibration humans (our calibration papers + RAID calibration abstracts); sentence-window
thresholds from 3-sentence windows of the same human paragraphs. Also prints the document-level calibration
(flagged paragraphs per human paper) and keeps the structure model and norms of the current model.json.
  make_or_model.py deberta-base-v2 deberta-base-v3
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from finetune import load_rows, predict, sentences  # noqa: E402
from aidet.classifier import load_classifier  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "aidet", "model.json")


def main():
    members = sys.argv[1:]
    k = len(members)
    rows = load_rows(raid=True)
    split = json.load(open(os.path.join(D, "split.json")))
    part = lambda r: r.get("part") or split.get(r["paper"])
    cal_h = [r for r in rows if part(r) == "calib" and r["y"] == 0]
    papers = [r for r in rows if part(r) in ("calib", "test") and r["y"] == 0 and r.get("kind") != "raid"]
    win = [" ".join(s[i:i + 3]) for r in cal_h for s in [sentences(r["text"])] for i in range(max(1, len(s) - 2))]
    old = json.load(open(OUT))
    model = {"type": "or", "members": [], "alpha": 0.01,
             "structure": old["structure"], "norms": old["norms"], "n_norm_papers": old.get("n_norm_papers")}
    flags = {}
    for m in members:
        clf, tok = load_classifier(os.path.join(os.path.dirname(__file__), "..", "aidet", "models", m))
        clf.cuda()
        sc = np.array(predict(clf, tok, [r["text"] for r in cal_h]))
        sw = np.array(predict(clf, tok, win))
        sp = np.array(predict(clf, tok, [r["text"] for r in papers]))
        t_ai, t_unc = float(np.quantile(sc, 1 - 0.01 / k)), float(np.quantile(sc, 1 - 0.05 / k))
        model["members"].append({"model": m, "thr_ai": t_ai, "thr_uncertain": t_unc,
                                 "thr_sentence": float(np.quantile(sw, 1 - 0.01 / k)),
                                 "thr_sentence_5": float(np.quantile(sw, 1 - 0.05 / k))})
        flags[m] = sp > t_ai
        del clf
    anyflag = np.any(list(flags.values()), axis=0)
    by = {}
    for r, f in zip(papers, anyflag):
        by.setdefault(r["paper"], []).append(f)
    shares = [(sum(v), len(v)) for v in by.values() if len(v) >= 5]
    print(f"document level, {len(shares)} human papers with >= 5 paragraphs:")
    for k_min, frac in ((1, 0.0), (2, 0.0), (3, 0.15), (3, 0.20), (3, 0.30)):
        print(f"  >= {k_min} flagged and >= {frac:.0%}: human papers flagged {np.mean([a >= k_min and a / n >= frac for a, n in shares]):.3f}")
    print(f"paragraph FPR on these human papers: {anyflag.mean():.4f}")
    json.dump(model, open(OUT, "w"), indent=1)
    print("wrote aidet/model.json:", json.dumps(model["members"], indent=1))


if __name__ == "__main__":
    main()
