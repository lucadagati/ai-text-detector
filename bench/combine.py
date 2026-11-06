"""Combine specialist detectors at a fixed overall false-positive budget.

Calibration humans = our calibration papers + RAID calibration abstracts; test = our test papers + RAID test.
Candidates: single models; OR rules (flag if any member exceeds its own alpha/k threshold, a union bound
that keeps the overall FPR <= alpha); logistic stacking. Reported on: our domain (Claude, AI rewrites),
RAID (clean / attacked), held-out abstracts (FPR) and manuscripts (share flagged).
  combine.py [--doc NAME=GLOB ...] [--models deberta-base-v2,deberta-base-v3] [--ext MAGE]
"""
import argparse, glob, itertools, json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from finetune import load_rows, predict  # noqa: E402
from external import EXT, ext_scores  # noqa: E402
from aidet.classifier import load_classifier  # noqa: E402
from aidet.textprep import latex_to_text, split_paragraphs  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
ALPHA = 0.01


def cached_scores(name, keys, texts, fn):
    path = os.path.join(D, f"comb_{name}.jsonl")
    cache = {json.loads(l)["k"]: json.loads(l)["s"] for l in open(path)} if os.path.exists(path) else {}
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        print("scoring", name, len(todo), flush=True)
        for i, s in zip(todo, fn([texts[i] for i in todo])):
            cache[keys[i]] = s
        with open(path, "w") as f:
            for k, s in cache.items():
                f.write(json.dumps({"k": k, "s": s}) + "\n")
    return np.array([cache[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[])
    ap.add_argument("--models", default="deberta-base-v2,deberta-base-v3")
    ap.add_argument("--ext", default="MAGE")
    a = ap.parse_args()
    rows = load_rows(raid=True)
    split = json.load(open(os.path.join(D, "split.json")))
    part = lambda r: r.get("part") or split.get(r["paper"])
    items = [dict(r, set=part(r)) for r in rows if part(r) in ("calib", "test")]
    items += [{"id": f"abs{i}", "y": 0, "text": json.loads(l)["text"], "src": "human|heldout-abstract", "set": "heldout"}
              for i, l in enumerate(open(os.path.join(D, "abstracts_pre2020.jsonl")))]
    for spec in a.doc:
        name, pat = spec.split("=", 1)
        ps = [p for f in sorted(glob.glob(pat)) for p in split_paragraphs(open(f).read(), 40, latex=True)]
        items += [{"id": f"{name}{i}", "y": -1, "text": p, "src": name, "set": name} for i, p in enumerate(ps)]
    keys = [f"{it['set']}|{it['id']}" for it in items]
    texts = [" ".join(latex_to_text(it["text"]).split()) for it in items]

    S = {}
    for mname in a.models.split(","):
        def fn(tx, mname=mname):
            m, tok = load_classifier(os.path.join(os.path.dirname(__file__), "..", "aidet", "models", mname))
            m.cuda()
            out = predict(m, tok, tx)
            del m
            return [float(x) for x in out]
        S[mname] = cached_scores(mname, keys, texts, fn)
    for e in [x for x in a.ext.split(",") if x]:
        hub, maxlen = EXT[e]
        raw = [it["text"] for it in items]  # external detectors get the text as written
        S[e] = cached_scores(e, keys, raw, lambda tx: ext_scores(hub, maxlen, tx))

    st = np.array([it["set"] for it in items]); y = np.array([it["y"] for it in items])
    src = np.array([it.get("src", "") for it in items])
    cal, te = st == "calib", st == "test"
    for k in S:  # orient on calibration
        if roc_auc_score(y[cal], S[k][cal]) < 0.5:
            S[k] = -S[k]

    def thr(s, alpha):
        return np.quantile(s[cal & (y == 0)], 1 - alpha)

    def flags_or(members):
        alpha = ALPHA / len(members)
        return np.any([S[m] > thr(S[m], alpha) for m in members], axis=0)

    def flags_stack(members):
        X = np.stack([S[m] for m in members], 1)
        clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X[cal], y[cal])
        z = clf.decision_function(X)
        return z > np.quantile(z[cal & (y == 0)], 1 - ALPHA)

    cands = {f"{m}": (S[m] > thr(S[m], ALPHA)) for m in S}
    names = list(S)
    for k in (2, 3):
        for c in itertools.combinations(names, k):
            cands["OR(" + "+".join(c) + ")"] = flags_or(c)
            cands["STACK(" + "+".join(c) + ")"] = flags_stack(c)

    ours = te & np.array([not x.startswith("raid") and x != "human" for x in src])
    groups = {
        "FP ours": te & (y == 0) & ~np.array([s.startswith("raid") for s in src]) & (src != "human|raid"),
        "FP RAID": te & (y == 0) & np.array([it.get("kind") == "raid" for it in items]),
        "FP heldout": st == "heldout",
        "Claude": te & np.array([x.startswith("claude") for x in src]),
        "rewrites": te & np.array([x.split("|")[-1] in ("polish", "revise", "multipass") and not x.startswith("raid") for x in src]),
        "our AI": ours & (y == 1),
        "RAID clean": te & np.array([x.startswith("raid") and x.endswith("|none") for x in src]),
        "RAID attack": te & np.array([x.startswith("raid") and not x.endswith("|none") for x in src]) & (y == 1),
    }
    docs = [d.split("=", 1)[0] for d in a.doc]
    print(f"\n{'combination':44s}" + "".join(f"{g:>12s}" for g in groups) + "".join(f"{d:>10s}" for d in docs))
    for name, f in cands.items():
        print(f"{name:44s}" + "".join(f"{np.mean(f[m]):12.3f}" for m in groups.values())
              + "".join(f"{np.mean(f[st == d]):10.0%}" for d in docs))


if __name__ == "__main__":
    main()
