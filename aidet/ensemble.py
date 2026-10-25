"""Ensemble decision (weights, thresholds and structure model from bench/stack.py -> aidet/model.json).

Per paragraph: the signals chosen by stack.py are combined into one score; `ai` above the threshold that
flags 1% of human calibration paragraphs, `uncertain` between the 1% and 5% thresholds.
Per sentence: the fine-tuned classifier scores windows of 3 sentences; each sentence gets the mean of the
windows that contain it (for highlighting only).
Per document: the structural traits farthest from the human norm, as an explanation.
"""
import json, math, os, re, threading

import numpy as np

from .textprep import split_paragraphs

HERE = os.path.dirname(__file__)
MODEL = json.load(open(os.path.join(HERE, "model.json")))
HF_REPO = os.environ.get("AIDET_HF_REPO", "Lukkinen/ai-text-detector")


def model_dir(name):
    """Local folder of a trained classifier; downloaded from the Hugging Face Hub on first use."""
    path = os.path.join(HERE, "models", name)
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, allow_patterns=[f"{name}/*"], local_dir=os.path.join(HERE, "models"))
    return path
_thr = os.path.join(HERE, "models", MODEL.get("ft_model", "-"), "thresholds.json")
if MODEL.get("type") != "or" and os.path.exists(_thr):
    _t = json.load(open(_thr))
    MODEL["thr_sentence"], MODEL["thr_sentence_5"] = _t["thr_sentence_1fpr"], _t["thr_sentence_5fpr"]


_dn = os.path.join(HERE, "docnorm.json")
DOCNORM = json.load(open(_dn)) if os.path.exists(_dn) else None


def structure_distance(paras):
    """Document-level distance of the sentence-structure profile from the human norm (bench/doc_structure.py).

    The text is cut into chunks of K paragraphs (K = size of the calibration pseudo-documents); the median chunk
    distance is compared with the 95% / 99% quantiles of human pseudo-documents. Independent of the classifier,
    so a text revised until the classifier passes it can still show up here."""
    from .syntax import _agg, sentence_records
    dn = DOCNORM
    if not dn or len(paras) < dn["k"] // 2:
        return None
    recs = [sentence_records(p) for p in paras]
    mu, P, k = np.array(dn["mean"]), np.array(dn["precision"]), dn["k"]
    ds = []
    for i in range(0, len(paras) - k // 2 + 1, k):
        r = [x for rs in recs[i:i + k] for x in rs]
        if len(r) < 8:
            continue
        f = _agg(r)
        v = np.array([f[key] for key in dn["keys"]]) - mu
        ds.append(float(np.sqrt(v @ P @ v)))
    if not ds:
        return None
    med = float(np.median(ds))
    level = "high" if med > dn["p99"] * 1.25 else ("elevated" if med > dn["p99"] else ("borderline" if med > dn["p95"] else "typical"))
    return {"distance": round(med, 2), "p95": round(dn["p95"], 2), "p99": round(dn["p99"], 2), "level": level,
            "chunks": [round(d, 2) for d in ds]}


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t) if s.strip()]


class Detector:
    def __init__(self):
        self.lock = threading.Lock()
        self.m = None

    # ------------------------------------------------------------------ models
    def _load(self):
        import torch
        from .classifier import load_classifier
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = {"torch": torch}
        if MODEL.get("type") == "or":
            m["members"] = []
            for mem in MODEL["members"]:
                clf, tok = load_classifier(model_dir(mem["model"]))
                m["members"].append((clf.to(dev).eval(), tok, mem))
            self.m = m
            return
        sig = MODEL["signals"]
        m["ft"], m["tok"] = load_classifier(model_dir(MODEL["ft_model"]))
        m["ft"].to(dev).eval()
        if "desklib" in sig:
            from .detectors import Desklib
            m["desklib"] = Desklib()
        if "fakespot" in sig:
            from .detectors import Fakespot
            m["fakespot"] = Fakespot()
        if "binoculars" in sig:
            from .detectors import Binoculars
            m["binoculars"] = Binoculars("falcon")
        self.m = m

    @property
    def loaded(self):
        return self.m is not None

    def unload(self):
        with self.lock:
            if self.m is not None:
                torch = self.m["torch"]
                self.m = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _ft(self, texts, model=None, tok=None):
        torch = self.m["torch"]
        model, tok = model or self.m["ft"], tok or self.m["tok"]
        dev = next(model.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i + 32], truncation=True, max_length=384, padding=True, return_tensors="pt").to(dev)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    out += model(enc["input_ids"], enc["attention_mask"]).float().view(-1).tolist()
        return out

    # ------------------------------------------------------------------ scoring
    def _structure(self, feats):
        s = MODEL["structure"]
        x = (np.array([feats[k] for k in s["keys"]]) - np.array(s["mean"])) / np.array(s["scale"])
        return float(x @ np.array(s["coef"]) + s["intercept"])

    def analyze(self, text, min_words=40, latex=None):
        from .syntax import paragraph_features, profile
        paras, skipped = split_paragraphs(text, min_words, latex, with_skipped=True)
        with self.lock:
            if self.m is None:
                self._load()
            if MODEL.get("type") == "or":
                out = self._analyze_or(paras)
            else:
                out = self._analyze_single(paras)
        return self._summary(paras, skipped, out)

    def _analyze_or(self, paras):
        """normalized scores: >= 1 above some member's AI threshold, >= 0 above some member's uncertain one"""
        if not paras:
            return []
        per = []
        for clf, tok, mem in self.m["members"]:
            s = np.array(self._ft(paras, clf, tok))
            per.append((s - mem["thr_uncertain"]) / (mem["thr_ai"] - mem["thr_uncertain"]))
        score = np.max(per, axis=0)
        out = []
        for j, p in enumerate(paras):
            ss = sentences(p)
            win = [" ".join(ss[i:i + 3]) for i in range(max(1, len(ss) - 2))]
            wsc = []
            for clf, tok, mem in self.m["members"]:
                w = np.array(self._ft(win, clf, tok))
                wsc.append((w - mem["thr_sentence_5"]) / (mem["thr_sentence"] - mem["thr_sentence_5"]))
            wmax = np.max(wsc, axis=0)
            sent = []
            for k, s in enumerate(ss):
                cover = [wmax[i] for i in range(len(win)) if i <= k < i + 3] or [wmax[0]]
                sent.append({"text": s, "score": float(np.mean(cover))})
            z = float(score[j])
            label = "ai" if z >= 1 else ("uncertain" if z >= 0 else "human")
            out.append({"text": p, "words": len(p.split()), "score": z, "label": label,
                        "signals": {mem["model"]: round(float(v[j]), 3) for (_, _, mem), v in zip(self.m["members"], per)},
                        "sentences": sent})
        return out

    def _analyze_single(self, paras):
        from .syntax import paragraph_features
        if True:
            ft = self._ft(paras) if paras else []
            out = []
            for p, f_ft in zip(paras, ft):
                sig = {"ft": f_ft}
                if "desklib" in self.m:
                    sig["desklib"] = _logit(self.m["desklib"](p))
                if "fakespot" in self.m:
                    sig["fakespot"] = _logit(self.m["fakespot"](p))
                if "binoculars" in self.m:
                    sig["binoculars"] = -self.m["binoculars"](p)
                if "structure" in MODEL["signals"]:
                    sf = paragraph_features(p)
                    sig["structure"] = self._structure(sf) if sf else 0.0
                z = sum(c * sig[k] for c, k in zip(MODEL["coef"], MODEL["signals"])) + MODEL["intercept"]
                label = "ai" if z >= MODEL["thr_1fpr"] else ("uncertain" if z >= MODEL["thr_5fpr"] else "human")
                # sentence highlighting: windows of 3 sentences
                ss = sentences(p)
                win = [" ".join(ss[i:i + 3]) for i in range(max(1, len(ss) - 2))]
                ws = self._ft(win)
                sent = []
                for j, s in enumerate(ss):
                    cover = [ws[i] for i in range(len(win)) if i <= j < i + 3] or ws[:1]
                    sent.append({"text": s, "score": float(np.mean(cover))})
                out.append({"text": p, "words": len(p.split()), "score": float(z), "label": label,
                            "signals": {k: round(v, 3) for k, v in sig.items()}, "sentences": sent})
        return out

    def _summary(self, paras, skipped, out):
        from .syntax import profile
        words = sum(o["words"] for o in out) or 1
        share = {k: sum(o["words"] for o in out if o["label"] == k) / words for k in ("ai", "uncertain", "human")}
        # document rules calibrated on 90 human calibration/test papers (bench/stack.py): ">= 3 flagged paragraphs
        # and >= 15% flagged" fires on 1 of 90 human papers; isolated flagged paragraphs occur in ~8% of them.
        n, k = len(out), sum(o["label"] == "ai" for o in out)
        if not out:
            verdict = "too short"
        elif (k >= 3 and k / n >= 0.30) or (n < 5 and k / n >= 0.5):
            verdict = "likely AI-generated"
        elif (k >= 3 and k / n >= 0.15) or (n < 5 and k >= 1):
            verdict = "mixed: parts likely AI-generated"
        elif k >= 1:
            verdict = "likely human (isolated flagged paragraphs, inconclusive)"
        else:
            verdict = "likely human"
        sdist = structure_distance([o["text"] for o in out])
        if sdist and sdist["level"] in ("elevated", "high") and verdict.startswith("likely human"):
            verdict = ("likely human by the classifier, but sentence structure is far from human papers "
                       f"(distance {sdist['distance']} vs p99 {sdist['p99']}): possible detector-guided revision")
        traits = []
        if paras and "norms" in MODEL:
            prof, n = profile(paras)
            for k, (mu, sd) in MODEL["norms"].items():
                if k in prof and sd > 0 and "entropy" not in k:  # entropy grows with text length
                    traits.append({"trait": k, "value": round(prof[k], 2), "human_mean": round(mu, 2),
                                   "z": round((prof[k] - mu) / sd, 2)})
            traits.sort(key=lambda t: -abs(t["z"]))
        orm = MODEL.get("type") == "or"
        return {"verdict": verdict, "share_words": share, "n_paragraphs": len(out), "n_skipped": skipped, "paragraphs": out,
                "structure_traits": traits[:8], "structure_distance": sdist,
                "sentence_threshold": 1.0 if orm else MODEL.get("thr_sentence", 0.0),
                "sentence_threshold_5": 0.0 if orm else MODEL.get("thr_sentence_5", 0.0),
                "thresholds": {"ai": 1.0, "uncertain": 0.0} if orm else {"ai": MODEL["thr_1fpr"], "uncertain": MODEL["thr_5fpr"]}}
