"""Compare OR ensembles of the fine-tuned classifiers on the held-out test papers (all modes, incl. adversarial).

  eval_or.py v2,v3,v4 [--ext NAME=FILE.tex ...]
Per member the score is normalised with thresholds at alpha/k FPR of the calibration humans; a paragraph is flagged
if any member flags it. Prints human FPR and detection per generator group / mode for each subset in --subsets.
Scores are cached in bench/data/or_scores_<model>.json.
"""
import argparse, itertools, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from finetune import load_rows, predict  # noqa: E402
from aidet.classifier import load_classifier  # noqa: E402
from aidet.textprep import split_paragraphs  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def scores(name, rows, ext):
    path = os.path.join(D, f"or_scores_{name}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    todo = [r for r in rows if r["id"] not in cache] + [e for e in ext if e["id"] not in cache]
    if todo:
        clf, tok = load_classifier(os.path.join(ROOT, "aidet", "models", f"deberta-base-{name}"))
        clf.cuda()
        for r, s in zip(todo, predict(clf, tok, [r["text"] for r in todo])):
            cache[r["id"]] = float(s)
        json.dump(cache, open(path, "w"))
        del clf
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members")
    ap.add_argument("--ext", action="append", default=[])
    a = ap.parse_args()
    names = a.members.split(",")
    rows = load_rows(raid=True)
    split = json.load(open(os.path.join(D, "split.json")))
    part = lambda r: r.get("part") or split.get(r["paper"])
    rows = [r for r in rows if part(r) in ("calib", "test")]
    ext = []
    for spec in a.ext:
        n, f = spec.split("=", 1)
        for i, p in enumerate(split_paragraphs(open(f).read(), 40, latex=True)):
            ext.append({"id": f"ext:{n}#{i}", "text": " ".join(p.split()), "name": n})
    S = {n: scores(n, rows, ext) for n in names}
    cal = [r for r in rows if part(r) == "calib" and r["y"] == 0]
    te = [r for r in rows if part(r) == "test"]
    y = np.array([r["y"] for r in te]); src = np.array([r["src"] for r in te])
    kind = np.array([r["kind"] for r in te])
    def flagged(sub, alpha, items):
        k = len(sub)
        out = np.zeros(len(items), bool)
        for n in sub:
            t = np.quantile([S[n][r["id"]] for r in cal], 1 - alpha / k)
            out |= np.array([S[n][r["id"]] for r in items]) > t
        return out
    groups = {"claude": lambda s: s.startswith("claude"), "open LLMs": lambda s: s.startswith("ollama"),
              "mode adv": lambda s: s.endswith("|adv"), "mode adv2": lambda s: s.endswith("|adv2"), "mode polish": lambda s: s.endswith("|polish"),
              "mode revise": lambda s: s.endswith("|revise"), "mode multipass": lambda s: s.endswith("|multipass"),
              "RAID": lambda s: s.startswith(("raid", "gpt", "mistral", "cohere", "llama", "mpt", "chatgpt"))}
    for r in range(1, len(names) + 1):
        for sub in itertools.combinations(names, r):
            for alpha in (0.01,):
                f = flagged(sub, alpha, te)
                line = f"{'+'.join(sub):10s} a={alpha}: FPR body {f[(y==0)&(kind=='body')].mean():.3f} abs {f[(y==0)&(kind=='abstract')].mean():.3f} raid {f[(y==0)&(kind=='raid')].mean():.3f} | "
                line += " ".join(f"{g} {f[(y==1)&np.array([fn(s) for s in src])].mean():.2f}" for g, fn in groups.items())
                ev = [r for r in te if r["y"] == 1 and r["src"].endswith("|adv")]
                print(line, flush=True)
        if ext:
            for sub in itertools.combinations(names, r):
                for n in sorted({e["name"] for e in ext}):
                    its = [e for e in ext if e["name"] == n]
                    print(f"   {'+'.join(sub):10s} {n}: {flagged(sub, 0.01, its).mean():.0%} of {len(its)} paragraphs flagged")
if __name__ == "__main__":
    main()
