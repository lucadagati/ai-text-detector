"""Partially edited text: hybrid paragraphs built from held-out test papers.

AI paragraph (Claude, modes free/human/polish/revise) with a fraction `keep` of its sentences left in place and the
others replaced by sentences of human paragraphs of the same paper (a person editing an LLM draft); and the reverse,
a human paragraph with a fraction of its sentences replaced by AI sentences of the same paper. Scored by the running
service (AIDET_URL). Prints the share labelled ai, and ai-or-uncertain, per mixing level.
  eval_hybrid.py [--n 250]
"""
import argparse, json, os, random, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from finetune import norm, sentences  # noqa: E402
D = os.path.join(os.path.dirname(__file__), "data")
URL = os.environ.get("AIDET_URL", "http://127.0.0.1:8790")

def label(text):
    body = json.dumps({"text": text, "latex": False, "min_words": 20}).encode()
    p = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/api/detect", body, {"Content-Type": "application/json"}), timeout=300).read())["paragraphs"]
    return p[0]["label"] if p else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=250); a = ap.parse_args()
    split = json.load(open(os.path.join(D, "split.json"))); rnd = random.Random(7)
    hum, ai = {}, {}
    for l in open(os.path.join(D, "human.jsonl")):
        r = json.loads(l)
        if split.get(r["paper"]) == "test": hum.setdefault(r["paper"], []).append(sentences(norm(r["text"])))
    for l in open(os.path.join(D, "ai.jsonl")):
        r = json.loads(l)
        if split.get(r["paper"]) == "test" and r["gen"].startswith("claude") and r["mode"] in ("free", "human", "polish", "revise") and r.get("kind", "body") == "body":
            ai.setdefault(r["paper"], []).append(sentences(norm(r["text"])))
    pool_h = {p: [s for par in v for s in par] for p, v in hum.items()}
    pool_a = {p: [s for par in v for s in par] for p, v in ai.items()}
    def build(src, pool, keep):
        out = []
        idx = list(range(len(src))); rnd.shuffle(idx); kept = set(idx[:max(1, round(keep * len(src)))])
        for i, s in enumerate(src):
            out.append(s if i in kept else rnd.choice(pool))
        return " ".join(out)
    print(f"{'condition':58s} {'n':>4s} {'ai':>6s} {'ai+unc':>7s}")
    for keep in (1.0, 0.75, 0.5, 0.25):
        items = [(p, par) for p, v in ai.items() if p in pool_h for par in v if len(par) >= 4]
        rnd.shuffle(items); items = items[:a.n]; c = {"ai": 0, "uncertain": 0, "n": 0}
        for p, par in items:
            lab = label(build(par, pool_h[p], keep))
            if lab: c["n"] += 1; c["ai"] += lab == "ai"; c["uncertain"] += lab == "uncertain"
        print(f"{'AI paragraph, ' + format(keep, '.0%') + ' of its sentences kept, rest human':58s} {c['n']:4d} {c['ai']/c['n']:6.1%} {(c['ai']+c['uncertain'])/c['n']:7.1%}", flush=True)
    for g in (0.0, 0.25, 0.5):
        items = [(p, par) for p, v in hum.items() if p in pool_a for par in v if len(par) >= 4]
        rnd.shuffle(items); items = items[:a.n]; c = {"ai": 0, "uncertain": 0, "n": 0}
        for p, par in items:
            lab = label(build(par, pool_a[p], 1 - g) if g else " ".join(par))
            if lab: c["n"] += 1; c["ai"] += lab == "ai"; c["uncertain"] += lab == "uncertain"
        print(f"{'human paragraph, ' + format(g, '.0%') + ' of its sentences replaced by AI':58s} {c['n']:4d} {c['ai']/c['n']:6.1%} {(c['ai']+c['uncertain'])/c['n']:7.1%}", flush=True)
main()
