"""Compare our detector with independent open detectors from other groups, on the same texts.

  external.py [--doc NAME=GLOB ...]
Sets: test papers of our benchmark, the held-out pre-2020 arXiv abstracts, the given manuscripts and,
if present, bench/data/raid_abstracts.jsonl (a neutral public benchmark, see fetch_raid.py).
Each detector's orientation and thresholds are fixed on OUR calibration papers' human paragraphs (1% / 5% FPR),
so every detector is judged at the same false-positive budget. Scores are cached in bench/data/ext_<name>.jsonl.
"""
import argparse, glob, json, os, sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
from finetune import load_rows, predict  # noqa: E402
from aidet.classifier import load_classifier  # noqa: E402
from aidet.textprep import latex_to_text, split_paragraphs  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
EXT = {  # name: (hub id, max tokens)
    "RADAR": ("TrustSafeAI/RADAR-Vicuna-7B", 512),
    "MAGE": ("yaful/MAGE", 1024),
    "academic-RoBERTa": ("andreas122001/roberta-academic-detector", 512),
    "ChatGPT-RoBERTa": ("Hello-SimpleAI/chatgpt-detector-roberta", 512),
    "OpenAI-RoBERTa": ("openai-community/roberta-large-openai-detector", 512),
}


@torch.no_grad()
def ext_scores(hub, maxlen, texts, bs=16):
    tok = AutoTokenizer.from_pretrained(hub)
    m = AutoModelForSequenceClassification.from_pretrained(hub).cuda().eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
        lp = torch.log_softmax(m(**enc).logits.float(), -1)
        out += (lp[:, 1] - lp[:, 0]).tolist()  # log-odds of label 1; orientation fixed later on calibration data
    del m
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="append", default=[])
    a = ap.parse_args()
    rows = load_rows()
    split = json.load(open(os.path.join(D, "split.json")))
    sets = {"calib": [r for r in rows if split.get(r["paper"]) == "calib"],
            "test": [r for r in rows if split.get(r["paper"]) == "test"]}
    sets["heldout-abstracts"] = [{"id": f"abs{i}", "y": 0, "text": json.loads(l)["text"], "src": "human"}
                                 for i, l in enumerate(open(os.path.join(D, "abstracts_pre2020.jsonl")))]
    raid = os.path.join(D, "raid_abstracts.jsonl")
    if os.path.exists(raid):
        sets["RAID-abstracts"] = [r for r in map(json.loads, open(raid)) if r.get("part", "test") == "test"]
    for spec in a.doc:
        name, pat = spec.split("=", 1)
        sets[name] = [{"id": f"{name}{i}", "y": None, "text": p, "src": name}
                      for i, p in enumerate(q for f in sorted(glob.glob(pat)) for q in split_paragraphs(open(f).read(), 40, latex=True))]
    allrows = {f"{s}|{r['id']}": r for s, rs in sets.items() for r in rs}
    keys = list(allrows)
    texts = [allrows[k]["text"] for k in keys]

    scores = {}
    for name, (hub, maxlen) in EXT.items():
        path = os.path.join(D, f"ext_{name}.jsonl")
        cache = {json.loads(l)["k"]: json.loads(l)["s"] for l in open(path)} if os.path.exists(path) else {}
        todo = [k for k in keys if k not in cache]
        if todo:
            print("scoring", name, len(todo), flush=True)
            for k, s in zip(todo, ext_scores(hub, maxlen, [allrows[k]["text"] for k in todo])):
                cache[k] = s
            with open(path, "w") as f:
                for k, s in cache.items():
                    f.write(json.dumps({"k": k, "s": s}) + "\n")
        scores[name] = np.array([cache[k] for k in keys])
    ap_model = os.environ.get("AIDET_MODEL", "deberta-base-v2")
    m, tok = load_classifier(os.path.join(os.path.dirname(__file__), "..", "aidet", "models", ap_model))
    m.cuda()
    # ours sees the text after its own preparation (Unicode/homoglyph normalization), as the service does
    ours_texts = [" ".join(latex_to_text(t).split()) for t in texts]
    scores["OURS (v2)"] = np.array(predict(m, tok, ours_texts))

    idx = {s: np.array([k.startswith(s + "|") for k in keys]) for s in sets}
    y = np.array([allrows[k]["y"] if allrows[k]["y"] is not None else -1 for k in keys])
    src = np.array([allrows[k].get("src", "") for k in keys])
    print(f"\n{'detector':18s} {'test AUROC':>10s} {'TPR@1%':>7s} {'Claude':>7s} {'polish':>7s} {'FP abs':>7s}"
          + ("  RAID AUROC  RAID TPR@1%" if "RAID-abstracts" in sets else "")
          + "".join(f" {d[:10]:>10s}" for d in sets if d not in ("calib", "test", "heldout-abstracts", "RAID-abstracts")))
    for name, s in scores.items():
        c = idx["calib"]
        if roc_auc_score(y[c], s[c]) < 0.5:
            s = -s  # orient on calibration data
        t1 = np.quantile(s[c & (y == 0)], 0.99)
        te = idx["test"]
        fp, tp, _ = roc_curve(y[te], s[te])
        cl = te & np.array([x.startswith("claude") for x in src])
        po = te & np.array([x.endswith("|polish") or x.endswith("|revise") or x.endswith("|multipass") for x in src])
        line = (f"{name:18s} {roc_auc_score(y[te], s[te]):10.3f} {np.interp(.01, fp, tp):7.3f} {np.mean(s[cl] > t1):7.3f} "
                f"{np.mean(s[po] > t1):7.3f} {np.mean(s[idx['heldout-abstracts']] > t1):7.3f}")
        if "RAID-abstracts" in sets:
            r = idx["RAID-abstracts"]
            fr, tr, _ = roc_curve(y[r], s[r])
            tr1 = np.quantile(s[r & (y == 0)], 0.99)  # RAID: its own 1%-FPR point (domain shift in our calibration)
            line += f"  {roc_auc_score(y[r], s[r]):10.3f}  {np.mean(s[r & (y == 1)] > tr1):11.3f}"
        for d in sets:
            if d not in ("calib", "test", "heldout-abstracts", "RAID-abstracts"):
                line += f" {np.mean(s[idx[d]] > t1):10.0%}"
        print(line)
    print("\ncolumns: TPR@1% and the per-source rates use each detector's threshold at 1% FPR on our calibration humans;"
          "\nClaude = Claude-written test paragraphs; polish = AI rewrites of human text (polish/revise/multipass);"
          "\nFP abs = false positives on 400 held-out pre-2020 arXiv abstracts; manuscript columns = share of paragraphs flagged.")


if __name__ == "__main__":
    main()
