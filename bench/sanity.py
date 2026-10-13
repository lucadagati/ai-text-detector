"""Shortcut checks for a fine-tuned classifier (does it learn style, or LaTeX-conversion artifacts?).

  sanity.py aidet/models/<name> [--doc NAME=GLOB ...]
(a) false positives on pre-2020 arXiv abstracts (plain text, never LaTeX-converted)
(b) test-set human paragraphs with vs without conversion artifacts (x placeholders, 'Section 3')
(c) artifact injection: AI test paragraphs with placeholders added; human ones with them removed
(d) share of paragraphs flagged in the given manuscripts
Threshold: 1% FPR on the calibration papers' human paragraphs.
"""
import argparse, glob, json, os, random, re, sys, time, urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.classifier import load_classifier  # noqa: E402
from aidet.textprep import split_paragraphs  # noqa: E402
from finetune import load_rows, predict  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
ABS = os.path.join(D, "abstracts_pre2020.jsonl")
ART = re.compile(r"\bx\b|Section 3")


def abstracts(n=400):
    if os.path.exists(ABS):
        return [json.loads(l)["text"] for l in open(ABS)]
    out = []
    for cat in ["cs.DC", "cs.RO", "cs.SE", "eess.SY", "cs.NI", "cs.CR", "cs.LG", "cs.AI"]:
        q = f"cat:{cat}+AND+submittedDate:[201501010000+TO+201912312359]"
        req = urllib.request.Request(f"https://export.arxiv.org/api/query?search_query={q}&start=500&max_results=60",
                                     headers={"User-Agent": "ai-detector-bench/1.1"})
        xml = urllib.request.urlopen(req, timeout=90).read().decode()
        out += [" ".join(s.split()) for s in re.findall(r"<summary>(.*?)</summary>", xml, re.S)]
        time.sleep(3)
    out = [a for a in out if len(a.split()) >= 60][:n]
    with open(ABS, "w") as f:
        for a in out:
            f.write(json.dumps({"text": a}) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--doc", action="append", default=[])
    a = ap.parse_args()
    model, tok = load_classifier(a.model)
    model.cuda()
    rows = load_rows()
    split = json.load(open(os.path.join(D, "split.json")))
    ca = [r for r in rows if split.get(r["paper"]) == "calib"]
    te = [r for r in rows if split.get(r["paper"]) == "test"]
    sc = predict(model, tok, [r["text"] for r in ca if r["y"] == 0])
    t1 = float(np.quantile(sc, 0.99))
    flag = lambda texts: float(np.mean(predict(model, tok, texts) > t1)) if texts else float("nan")
    ab = abstracts()
    print(f"(a) pre-2020 arXiv abstracts flagged as AI: {flag(ab):.3f}  (n={len(ab)}; target ~0.01)")
    h = [r["text"] for r in te if r["y"] == 0]
    ai = [r["text"] for r in te if r["y"] == 1]
    h_art = [t for t in h if ART.search(t)]; h_clean = [t for t in h if not ART.search(t)]
    ai_art = [t for t in ai if ART.search(t)]; ai_clean = [t for t in ai if not ART.search(t)]
    print(f"(b) human test flagged: with artifacts {flag(h_art):.3f} (n={len(h_art)}), without {flag(h_clean):.3f} (n={len(h_clean)})")
    print(f"    AI test detected:   with artifacts {flag(ai_art):.3f} (n={len(ai_art)}), without {flag(ai_clean):.3f} (n={len(ai_clean)})")
    rnd = random.Random(0)

    def inject(t):
        w = t.split()
        for _ in range(max(1, len(w) // 40)):
            w.insert(rnd.randrange(len(w)), rnd.choice(["x", "(x)", "x,", "Section 3"]))
        return " ".join(w)
    print(f"(c) AI clean detected {flag(ai_clean):.3f} -> with injected placeholders {flag([inject(t) for t in ai_clean]):.3f}")
    stripped = [re.sub(r"\s*\(?\bx\b\)?,?", "", t).replace("Section 3", "this section") for t in h_art]
    print(f"    human with artifacts flagged {flag(h_art):.3f} -> artifacts removed {flag(stripped):.3f}")
    for spec in a.doc:
        name, pat = spec.split("=", 1)
        ps = [p for f in sorted(glob.glob(pat)) for p in split_paragraphs(open(f).read(), 40, latex=True)]
        print(f"(d) {name}: {flag(ps):.0%} of {len(ps)} paragraphs flagged as AI")


if __name__ == "__main__":
    main()
