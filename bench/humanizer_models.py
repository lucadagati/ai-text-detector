"""Which model should rewrite? One rewrite per model on the same flagged paragraphs; reports what the humanizer needs.

  python bench/humanizer_models.py FILE.tex [FILE2.tex ...] --models sonnet haiku ollama:mistral-small:24b [--n 8] [-o out.json]

For each model: how many rewrites pass the guards (numbers/refs/math, academic lint, <=25% growth), the mean score drop of the accepted
ones (positive = lower AI score), how many reach `human`, mean growth and seconds per call. Needs the running service (AIDET_URL) and
the `claude` CLI (or Ollama for `ollama:NAME` models). Run local models one at a time: they share the GPU.
"""
import argparse, json, os, statistics, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import humanize as H  # noqa: E402
from academic_lint import violations  # noqa: E402
from score_tex import blocks  # noqa: E402


def paragraphs(files, n, max_math):
    out = []
    for f in files:
        latex = f.endswith(".tex")
        for line, p in blocks(open(f, encoding="utf-8").read()):
            if len(p.split()) < 60 or H.math_share(p) > max_math:
                continue
            s = H.score(p, latex)
            if s and s[0] != "human":
                out.append((f, line, p, s, latex))
    return out[:n]


def run(model, pars):
    rows = []
    for f, line, p, (label, sc, d), latex in pars:
        t = time.time()
        try:
            c = H.rewrite(H.build_prompt(p, d, None, latex), model)
        except Exception:  # noqa
            rows.append({"ok": False, "why": "error", "t": time.time() - t})
            continue
        dt = time.time() - t
        why = None
        if len(c.split()) > 1.25 * len(p.split()):
            why = "grew"
        elif H.invariants(c) != H.invariants(p):
            why = "invariants"
        elif violations(p, c):
            why = "lint"
        r = {"ok": why is None, "why": why, "t": dt, "grow": len(c.split()) / len(p.split())}
        if why is None:
            s2 = H.score(c, latex)
            r["delta"], r["human"] = sc - s2[1], s2[0] == "human"
        rows.append(r)
    ok = [r for r in rows if r["ok"]]
    return {"n": len(rows), "accepted": len(ok),
            "mean_drop": statistics.mean(r["delta"] for r in ok) if ok else None,
            "human": sum(r.get("human", False) for r in ok),
            "mean_seconds": statistics.mean(r["t"] for r in rows),
            "mean_growth": statistics.mean(r["grow"] for r in ok) if ok else None,
            "rejected": {k: sum(1 for r in rows if r.get("why") == k) for k in ("grew", "invariants", "lint", "error")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8, help="paragraphs per model")
    ap.add_argument("--max-math", type=float, default=0.30)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    pars = paragraphs(a.files, a.n, a.max_math)
    print("paragraphs", len(pars), flush=True)
    res = {}
    for m in a.models:
        res[m] = run(m, pars)
        print(m, json.dumps(res[m]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
