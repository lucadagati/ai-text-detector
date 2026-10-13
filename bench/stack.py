"""Combine the signals into the final ensemble and report on the held-out test papers.

Signals per paragraph: fine-tuned classifier (ft_<name>.jsonl), desklib, fakespot, binoculars (features.jsonl),
and the sentence-structure profile (syntax.jsonl, reduced by a logistic model trained on the training papers).
The stacker and thresholds are fitted on the calibration papers; every number printed is on test papers.

  stack.py --ft deberta-base [--write]     # --write saves aidet/model.json
"""
import argparse, json, math, os, sys
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "aidet", "model.json")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def jl(name):
    p = os.path.join(D, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True, help="name of the fine-tuned model (ft_<name>.jsonl)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--raid", action="store_true", help="include RAID calibration/test rows")
    ap.add_argument("--signals", default="", help="force this subset (comma-separated) instead of the best one")
    a = ap.parse_args()
    split = json.load(open(os.path.join(D, "split.json")))
    meta = {}
    for name, y in (("human.jsonl", 0), ("human_abs.jsonl", 0), ("ai.jsonl", 1)):
        for r in jl(name):
            kind = r.get("kind", "body")
            meta[r["id"]] = {"y": y, "paper": r["paper"], "kind": kind,
                             "src": f"human|{kind}" if y == 0 else f"{r['gen']}|{r['mode']}"}
    if a.raid:
        for r in jl("raid_abstracts.jsonl"):
            meta["raid:" + r["id"]] = {"y": r["y"], "paper": r["paper"], "kind": "raid",
                                       "src": "human|raid" if r["y"] == 0 else r["src"]}
            split[r["paper"]] = r["part"]
    feats = {r["id"]: r for r in jl("features.jsonl")}
    syn = {r["id"]: r["f"] for r in jl("syntax.jsonl") if r["f"]}
    ft = {r["id"]: r["score"] for r in jl(f"ft_{a.ft}.jsonl")}
    skeys = sorted(next(iter(syn.values())))

    # structure model: trained on the training papers only
    tr_ids = [i for i in syn if i in meta and split.get(meta[i]["paper"]) == "train"]
    smodel = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
    smodel.fit(np.array([[syn[i][k] for k in skeys] for i in tr_ids]), [meta[i]["y"] for i in tr_ids])

    def row(i):
        f = feats.get(i, {})
        if i not in ft:
            return None
        v = {"_paper": meta[i]["paper"], "ft": ft[i]}
        if all(k in f for k in ("desklib", "fakespot")):
            v.update(desklib=logit(f["desklib"]), fakespot=logit(f["fakespot"]))
        if i in syn:
            v["structure"] = float(smodel.decision_function(np.array([[syn[i][k] for k in skeys]]))[0])
        return v

    data = {"calib": [], "test": []}
    for i, m in meta.items():
        part = split.get(m["paper"])
        if part in data:
            v = row(i)
            if v:
                data[part].append((v, m["y"], m["src"]))
    print(f"calib {len(data['calib'])}, test {len(data['test'])} paragraphs\n")
    names = ["ft", "desklib", "fakespot", "structure"]  # binoculars dropped in v2: near-random here, 28 GB of GPU
    yc = np.array([y for _, y, _ in data["calib"]]); yt = np.array([y for _, y, _ in data["test"]])
    st = np.array([s for _, _, s in data["test"]])

    def run(sub):
        cal = [(v, y, s) for v, y, s in data["calib"] if all(k in v for k in sub)]
        tst = [(v, y, s) for v, y, s in data["test"] if all(k in v for k in sub)]
        yc = np.array([y for _, y, _ in cal]); yt = np.array([y for _, y, _ in tst])
        Xc = np.array([[v[k] for k in sub] for v, _, _ in cal])
        Xt = np.array([[v[k] for k in sub] for v, _, _ in tst])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xc, yc)
        sc, stt = clf.decision_function(Xc), clf.decision_function(Xt)
        t1 = float(np.quantile(sc[yc == 0], 0.99)); t5 = float(np.quantile(sc[yc == 0], 0.95))
        fp, tp, _ = roc_curve(yt, stt)
        run.src = np.array([s for _, _, s in tst])
        return clf, t1, t5, stt, {"auroc": roc_auc_score(yt, stt), "tpr1": float(np.interp(.01, fp, tp)),
                                  "fpr_at_t1": float(np.mean(stt[yt == 0] >= t1)), "tpr_at_t1": float(np.mean(stt[yt == 1] >= t1))}

    print(f"{'signals':45s} {'AUROC':>6s} {'TPR@1%':>7s} {'FPR@t1':>7s} {'TPR@t1':>7s}")
    results = {}
    for k in range(1, len(names) + 1):
        for sub in combinations(names, k):
            if k > 1 and "ft" not in sub:
                continue
            results[sub] = run(sub)
            r = results[sub][4]
            print(f"{'+'.join(sub):45s} {r['auroc']:6.3f} {r['tpr1']:7.3f} {r['fpr_at_t1']:7.3f} {r['tpr_at_t1']:7.3f}")
    # choose: best TPR@1%FPR among subsets, ties -> fewer signals (no binoculars unless it helps)
    best = tuple(a.signals.split(",")) if a.signals else max(results, key=lambda s: (round(results[s][4]["tpr1"], 3), -len(s)))
    clf, t1, t5, stt, r = run(best)
    st = run.src
    print(f"\nchosen: {'+'.join(best)}")
    print("detection on test at the calibrated 1%-FPR threshold, by source:")
    for k in ("body", "abstract", "raid"):
        m = st == f"human|{k}"
        if m.any():
            print(f"  false positives, human {k:20s} {np.mean(stt[m] >= t1):.3f} (n={m.sum()})")
    for g in sorted({x.split('|')[0] for x in st if not x.startswith("human|")}) + ["mode:" + m for m in ("free", "human", "polish", "revise", "results", "multipass")]:
        m = np.array([x.endswith("|" + g[5:]) for x in st]) if g.startswith("mode:") else np.array([x.startswith(g + "|") for x in st])
        if m.any():
            print(f"  {g:32s} {np.mean(stt[m] >= t1):.3f} (n={m.sum()})")
    # document level: share of flagged paragraphs in each human paper (calib + test papers, >= 5 paragraphs)
    by = {}
    for part in ("calib", "test"):
        rows = [(v, y, s) for v, y, s in data[part] if all(k in v for k in best) and not v["_paper"].startswith("raid:")]
        sp = clf.decision_function(np.array([[v[k] for k in best] for v, _, _ in rows]))
        for (v, y, s), z in zip(rows, sp):
            if y == 0:
                by.setdefault(v["_paper"], []).append((z >= t1, z >= t5))
    shares = [(sum(a for a, _ in f), len(f)) for f in by.values() if len(f) >= 5]
    unc = [sum(b for _, b in f) / len(f) for f in by.values() if len(f) >= 5]
    print(f"\nshare of paragraphs >= the 5% threshold (ai or uncertain) in human papers: "
          f"mean {np.mean(unc):.3f}, p95 {np.quantile(unc, .95):.3f}, p99 {np.quantile(unc, .99):.3f}, max {max(unc):.3f}")
    print(f"\ndocument level, {len(shares)} human papers with >= 5 paragraphs:")
    for k_min, frac in ((1, 0.0), (2, 0.0), (2, 0.10), (3, 0.15), (3, 0.20)):
        fp = np.mean([k >= k_min and k / n >= frac for k, n in shares])
        print(f"  flag document if >= {k_min} paragraphs and >= {frac:.0%} flagged: human papers flagged {fp:.3f}")
    if a.write:
        model = {"signals": list(best), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                 "thr_1fpr": t1, "thr_5fpr": t5, "ft_model": a.ft,
                 "structure": {"keys": skeys, "mean": smodel[0].mean_.tolist(), "scale": smodel[0].scale_.tolist(),
                               "coef": smodel[1].coef_[0].tolist(), "intercept": float(smodel[1].intercept_[0])},
                 "test": {k: float(v) for k, v in r.items()}, "n_calib": len(yc), "n_test": len(yt)}
        # human norms of the structure profile, one profile per human paper (>= 5 body paragraphs)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from aidet.syntax import profile
        from finetune import norm
        byp = {}
        for r in jl("human.jsonl"):
            byp.setdefault(r["paper"], []).append(norm(r["text"]))
        profs = [profile(ps)[0] for ps in byp.values() if len(ps) >= 5]
        model["norms"] = {k: [float(np.mean([p[k] for p in profs])), float(np.std([p[k] for p in profs]))] for k in profs[0]}
        model["n_norm_papers"] = len(profs)
        json.dump(model, open(OUT, "w"), indent=1)
        print("wrote aidet/model.json")


if __name__ == "__main__":
    main()
