"""Fine-tune a domain classifier (human vs AI paragraphs of academic CS text).

  finetune.py --base microsoft/deberta-v3-base --out aidet/models/deberta-base
  finetune.py --base desklib --out aidet/models/desklib-ft      # continue from desklib-large

Split by paper (bench/data/split.json, created once): 60% train, 15% calibration, 25% test.
Training also uses random windows of 2-5 consecutive sentences, so the model scores short passages
(used for sentence highlighting). Writes the model, and the scores of every calibration/test paragraph
to bench/data/ft_<name>.jsonl for the ensemble.
"""
import argparse, json, math, os, random, re, sys, time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.textprep import assessable, latex_to_text  # noqa: E402
from aidet.classifier import load_classifier, new_classifier  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def norm(t):
    return " ".join(latex_to_text(t.replace("**", "")).split())


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if s.strip()]


def load_rows(raid=False):
    rows, seen = [], set()
    if raid and os.path.exists(os.path.join(D, "raid_abstracts.jsonl")):
        for l in open(os.path.join(D, "raid_abstracts.jsonl")):
            r = json.loads(l)
            rows.append({"id": "raid:" + r["id"], "paper": r["paper"], "y": r["y"], "kind": "raid",
                         "text": " ".join(latex_to_text(r["text"]).split()), "src": r["src"], "part": r["part"]})
            seen.add("raid:" + r["id"])
    for name, y in (("human.jsonl", 0), ("human_abs.jsonl", 0), ("ai.jsonl", 1), ("ai_adv.jsonl", 1), ("ai_adv2.jsonl", 1), ("ai_adv3.jsonl", 1)):
        if not os.path.exists(os.path.join(D, name)):
            continue
        for l in open(os.path.join(D, name)):
            r = json.loads(l)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            if not assessable(norm(r["text"])):
                continue
            rows.append({"id": r["id"], "paper": r["paper"], "y": y, "text": norm(r["text"]), "kind": r.get("kind", "body"),
                         "src": "human" if y == 0 else f"{r['gen']}|{r['mode']}"})
    return rows


def get_split(rows):
    path = os.path.join(D, "split.json")
    split = json.load(open(path)) if os.path.exists(path) else {}
    new = sorted({r["paper"] for r in rows} - set(split))
    rnd = random.Random(len(split))
    for p in new:  # papers added later get assigned with the same proportions
        u = rnd.random()
        split[p] = "train" if u < 0.60 else ("calib" if u < 0.75 else "test")
    json.dump(split, open(path, "w"), indent=0)
    return split


def windows(rows, per_row, rnd):
    out = []
    for r in rows:
        s = sentences(r["text"])
        for _ in range(per_row):
            if len(s) < 3:
                break
            k = rnd.randint(2, min(5, len(s) - 1))
            i = rnd.randint(0, len(s) - k)
            w = " ".join(s[i:i + k])
            if len(w.split()) >= 20:
                out.append({**r, "text": w})
    return out


@torch.no_grad()
def predict(model, tok, texts, bs=32, max_len=384):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(enc["input_ids"], enc["attention_mask"]).float().view(-1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--windows", type=int, default=2, help="sentence windows per training paragraph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raid", action="store_true", help="add RAID abstracts (bench/fetch_raid.py) with their own split")
    ap.add_argument("--extra", default="", help="comma-separated jsonl files of extra AI rows (manuscript revisions), always train")
    ap.add_argument("--extra-rep", type=int, default=6, help="repeat extra rows this many times")
    ap.add_argument("--exclude", default="", help="generator prefixes left out of train/calib (unseen-generator test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rnd = random.Random(a.seed)

    rows = load_rows(raid=a.raid)
    split = get_split([r for r in rows if "part" not in r])
    for r in rows:
        if "part" in r:
            split[r["paper"]] = r["part"]
    ex = tuple(p for p in a.exclude.split(",") if p)
    unseen = lambda r: bool(ex) and r["y"] == 1 and r["src"].startswith(ex)
    tr = [r for r in rows if split[r["paper"]] == "train" and not unseen(r)]
    ca = [r for r in rows if split[r["paper"]] == "calib" and not unseen(r)]
    te = [r for r in rows if split[r["paper"]] == "test"]
    ex_rows = []
    for f in [x for x in a.extra.split(",") if x]:
        for l in open(f):
            r = json.loads(l)
            ex_rows.append({"id": r["id"], "paper": r["paper"], "y": 1, "kind": "body", "text": r["text"], "src": "ms|extra"})
    tr = tr + ex_rows * a.extra_rep
    train = tr + windows(tr, a.windows, rnd)
    print(f"train {len(tr)} (+windows {len(train) - len(tr)}), calib {len(ca)}, test {len(te)}", flush=True)

    model, tok = new_classifier(a.base)
    model.cuda()
    pos = sum(r["y"] for r in train) / len(train)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        model.train(); rnd.shuffle(train)
        for i in range(0, len(train), a.bs):
            b = train[i:i + a.bs]
            enc = tok([r["text"] for r in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([r["y"] for r in b], dtype=torch.float, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(enc["input_ids"], enc["attention_mask"]).float().view(-1)
            loss = lossf(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        s = predict(model, tok, [r["text"] for r in ca], max_len=a.max_len)
        auc = roc_auc_score([r["y"] for r in ca], s)
        print(f"epoch {ep + 1}: calib AUROC {auc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best = auc
            os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
            model.save_pretrained(os.path.join(ROOT, a.out)); tok.save_pretrained(os.path.join(ROOT, a.out))

    model, tok = load_classifier(os.path.join(ROOT, a.out))
    model.cuda()
    name = os.path.basename(a.out.rstrip("/"))
    # threshold for sentence highlighting: 1% of 3-sentence windows of human calibration paragraphs
    hw = [" ".join(s[i:i + 3]) for r in ca if r["y"] == 0 for s in [sentences(r["text"])] for i in range(max(1, len(s) - 2))]
    sw = predict(model, tok, hw, max_len=a.max_len)
    json.dump({"thr_sentence_1fpr": float(np.quantile(sw, 0.99)), "thr_sentence_5fpr": float(np.quantile(sw, 0.95)),
               "n_windows": len(hw)}, open(os.path.join(ROOT, a.out, "thresholds.json"), "w"), indent=1)
    with open(os.path.join(D, f"ft_{name}.jsonl"), "w") as f:
        for part, rs in (("calib", ca), ("test", te)):
            s = predict(model, tok, [r["text"] for r in rs], max_len=a.max_len)
            for r, v in zip(rs, s):
                f.write(json.dumps({"id": r["id"], "part": part, "score": float(v)}) + "\n")
            if part == "calib":
                calib_scores = s
            if part == "test":
                y = np.array([r["y"] for r in rs]); src = np.array([r["src"] for r in rs])
                fp, tp, _ = roc_curve(y, s)
                sc, yc = calib_scores, np.array([r["y"] for r in ca])
                t1 = np.quantile(sc[yc == 0], 0.99)  # threshold from calibration humans
                kinds = np.array([r["kind"] for r in rs])
                for k in ("body", "abstract", "raid"):
                    m = (y == 0) & (kinds == k)
                    if m.any():
                        print(f"  human {k:24s} false positives at calib 1%FPR {np.mean(s[m] > t1):.3f} (n={m.sum()})")
                print(f"TEST AUROC {roc_auc_score(y, s):.4f}  TPR@1%FPR {np.interp(.01, fp, tp):.3f}  TPR@5%FPR {np.interp(.05, fp, tp):.3f}")
                if ex:
                    m = np.array([x.startswith(ex) for x in src])
                    print(f"  UNSEEN generators {','.join(ex)}: detected at 1%FPR {np.mean(s[m] > t1):.3f} (n={m.sum()})")
                for g in sorted({x.split('|')[0] for x in src if x != 'human'}):
                    m = np.array([x.startswith(g) for x in src])
                    print(f"  {g:30s} detected at 1%FPR {np.mean(s[m] > t1):.3f} (n={m.sum()})")
                for md in ["free", "human", "polish", "revise", "results", "multipass", "adv", "adv2", "adv3"]:
                    m = np.array([x.endswith('|' + md) for x in src])
                    if m.any():
                        print(f"  mode {md:25s} detected at 1%FPR {np.mean(s[m] > t1):.3f} (n={m.sum()})")


if __name__ == "__main__":
    main()
