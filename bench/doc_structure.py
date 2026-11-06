"""Document-level structural distance from the human norm.

  doc_structure.py [--doc NAME=GLOB ...]
Builds pseudo-documents of K paragraphs (human: random draws from distinct papers; AI: random draws per
generator/mode), computes a Mahalanobis distance of each document's structure profile from the human mean
(covariance from train papers, shrunk), calibrates thresholds at 95% / 99% of held-out human documents, and
reports how many AI documents (and the given manuscripts, in chunks of K paragraphs) exceed them.
Writes the norms to aidet/docnorm.json.
"""
import argparse, glob, json, os, random, sys

import numpy as np
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.syntax import _agg, sentence_records, DOC_KEYS  # noqa: E402
from aidet.textprep import latex_to_text, split_paragraphs  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(D, "sent_records.jsonl")
K = 12


def norm(t):
    return " ".join(latex_to_text(t.replace("**", "")).split())


def records(items):
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l); cache[r["id"]] = r["r"]
    todo = [it for it in items if it["id"] not in cache]
    if todo:
        from aidet.syntax import nlp
        with open(CACHE, "a") as fh:
            for it, doc in zip(todo, nlp().pipe([it["text"] for it in todo], batch_size=64, n_process=16)):
                from aidet.syntax import records_from_doc
                cache[it["id"]] = records_from_doc(doc)
                fh.write(json.dumps({"id": it["id"], "r": cache[it["id"]]}) + "\n")
    for it in items:
        it["r"] = cache[it["id"]]
    return items


def vec(items):
    recs = [r for it in items for r in it["r"]]
    if len(recs) < 8:
        return None
    f = _agg(recs)
    return np.array([f[k] for k in DOC_KEYS])


def pseudo_docs(items, n, rng, k=K):
    return [vec(rng.sample(items, k)) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[])
    a = ap.parse_args()
    split = json.load(open(os.path.join(D, "split.json")))
    H, A = [], []
    for n in ("human.jsonl", "human_abs.jsonl"):
        for l in open(os.path.join(D, n)):
            r = json.loads(l)
            if n == "human.jsonl":
                H.append({"id": r["id"], "paper": r["paper"], "text": norm(r["text"])})
    for l in open(os.path.join(D, "ai.jsonl")):
        r = json.loads(l)
        A.append({"id": r["id"], "paper": r["paper"], "gen": r["gen"], "mode": r["mode"], "text": norm(r["text"])})
    docs = {}
    for spec in a.doc:
        name, pat = spec.split("=", 1)
        its = []
        for f in sorted(glob.glob(pat)):
            for i, p in enumerate(split_paragraphs(open(f).read(), 40, latex=True)):
                its.append({"id": f"{name}:{os.path.basename(f)}#{i}", "text": p})
        docs[name] = its
    records(H + A + [i for v in docs.values() for i in v])
    rng = random.Random(0)
    H = [h for h in H if h["r"]]
    htr = [h for h in H if split.get(h["paper"]) == "train"]
    hca = [h for h in H if split.get(h["paper"]) == "calib"]
    hte = [h for h in H if split.get(h["paper"]) == "test"]
    Xtr = np.array([v for v in pseudo_docs(htr, 3000, rng) if v is not None])
    mu = Xtr.mean(0)
    lw = LedoitWolf().fit(Xtr)
    P = lw.precision_
    dist = lambda v: float(np.sqrt((v - mu) @ P @ (v - mu)))
    dc = np.array([dist(v) for v in pseudo_docs(hca, 600, rng) if v is not None])
    dt = np.array([dist(v) for v in pseudo_docs(hte, 600, rng) if v is not None])
    t95, t99 = np.quantile(dc, 0.95), np.quantile(dc, 0.99)
    print(f"calib-only thresholds: p95 {t95:.2f} p99 {t99:.2f}")
    t95, t99 = np.quantile(np.r_[dc, dt], 0.95), np.quantile(np.r_[dc, dt], 0.99)  # final: calib + test pooled
    print(f"human calib: median {np.median(dc):.2f} p95 {t95:.2f} p99 {t99:.2f}; human test: median {np.median(dt):.2f} "
          f"> p95 {np.mean(dt > t95):.1%}, > p99 {np.mean(dt > t99):.1%}")
    groups = {}
    for x in A:
        if x["r"]:
            groups.setdefault(("claude" if x["gen"].startswith("claude") else "other", x["mode"]), []).append(x)
    print("\nAI pseudo-docs (share above p95 / p99, median distance):")
    for (g, m), its in sorted(groups.items()):
        if len(its) < 2 * K:
            continue
        d = np.array([dist(v) for v in pseudo_docs(its, 300, rng) if v is not None])
        print(f"  {g:7s} {m:10s} n={len(its):4d}  >p95 {np.mean(d > t95):5.1%}  >p99 {np.mean(d > t99):5.1%}  median {np.median(d):5.2f}")
    for name, its in docs.items():
        its = [i for i in its if i["r"]]
        ds = [dist(v) for v in (vec(its[i:i + K]) for i in range(0, len(its) - K // 2, K)) if v is not None]
        whole = dist(vec(its)) if vec(its) is not None else float("nan")
        print(f"\n{name}: {len(its)} paragraphs, chunk distances {np.round(ds, 2).tolist()}, median {np.median(ds):.2f}, "
              f"whole-doc {whole:.2f}")
    json.dump({"keys": DOC_KEYS, "mean": mu.tolist(), "precision": P.tolist(), "k": K, "p95": float(t95), "p99": float(t99)},
              open(os.path.join(os.path.dirname(__file__), "..", "aidet", "docnorm.json"), "w"))


if __name__ == "__main__":
    main()
