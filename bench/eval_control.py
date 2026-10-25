"""False-positive control on modern-topic human text (arXiv 2021, before ChatGPT; never trained on).

  eval_control.py v2,v3,adv,xa,xb [--file bench/data/human_2021.jsonl]
Same thresholds as eval_or.py (alpha/k FPR of the calibration humans); prints the share of control paragraphs flagged
for every subset of the given members, and the share of papers with >= 2 flagged paragraphs.
"""
import argparse, itertools, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from eval_or import scores  # noqa: E402
from finetune import load_rows, norm  # noqa: E402
from aidet.textprep import assessable  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members")
    ap.add_argument("--file", default=os.path.join(D, "human_2021.jsonl"))
    a = ap.parse_args()
    names = a.members.split(",")
    ctl = []
    for l in open(a.file):
        r = json.loads(l)
        t = norm(r["text"])
        if assessable(t):
            ctl.append({"id": "ctl:" + r["id"], "paper": r["paper"], "text": t})
    rows = load_rows(raid=True)
    split = json.load(open(os.path.join(D, "split.json")))
    part = lambda r: r.get("part") or split.get(r["paper"])
    cal = [r for r in rows if part(r) == "calib" and r["y"] == 0]
    S = {n: scores(n, cal, ctl) for n in names}
    print(f"{len(ctl)} control paragraphs from {len({c['paper'] for c in ctl})} papers")
    for k in range(1, len(names) + 1):
        for sub in itertools.combinations(names, k):
            f = np.zeros(len(ctl), bool)
            for n in sub:
                t = np.quantile([S[n][r["id"]] for r in cal], 1 - 0.01 / len(sub))
                f |= np.array([S[n][c["id"]] for c in ctl]) > t
            by = {}
            for c, x in zip(ctl, f):
                by.setdefault(c["paper"], []).append(x)
            print(f"  {'+'.join(sub):16s} paragraphs flagged {f.mean():.3f}   papers with >=2 flagged {np.mean([sum(v) >= 2 for v in by.values()]):.3f}")


main()
