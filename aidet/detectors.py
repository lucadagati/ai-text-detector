"""The three detectors of the ensemble. Each returns one feature per text.

binoculars  zero-shot, Hans et al. (ICML 2024): perplexity / cross-perplexity of
            two related LMs (Qwen2.5-3B base as observer, -Instruct as performer).
            Lower = more machine-like.
desklib     supervised DeBERTa-v3-large classifier (desklib/ai-text-detector-v1.01,
            top of the RAID leaderboard in 2025). P(AI).
fakespot    supervised RoBERTa-base classifier (fakespot-ai/roberta-base-ai-text-detection-v1). P(AI).
"""
import threading
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from .classifier import MeanPoolClassifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32


PAIRS = {"falcon": ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),   # pair of the original paper
         "qwen": ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct")}


class Binoculars:
    name = "binoculars"

    def __init__(self, pair="falcon"):
        self.OBS, self.PERF = PAIRS[pair]
        self.name = "binoculars" if pair == "falcon" else f"binoculars_{pair}"
        self.tok = AutoTokenizer.from_pretrained(self.OBS)
        self.obs = AutoModelForCausalLM.from_pretrained(self.OBS, torch_dtype=DT).to(DEV).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(self.PERF, torch_dtype=DT).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids[:, :1024].to(DEV)
        lo = self.obs(ids).logits[0, :-1].float()
        lp = self.perf(ids).logits[0, :-1].float()
        ppl = nn.functional.cross_entropy(lp, ids[0, 1:])
        xppl = -(torch.softmax(lo, -1) * torch.log_softmax(lp, -1)).sum(-1).mean()
        return (ppl / xppl).item()


class Desklib:
    name = "desklib"
    ID = "desklib/ai-text-detector-v1.01"

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.ID)
        self.m = MeanPoolClassifier.from_pretrained(self.ID).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, text):
        enc = self.tok(text, truncation=True, max_length=768, return_tensors="pt").to(DEV)
        return torch.sigmoid(self.m(enc["input_ids"], enc["attention_mask"])).item()


class Fakespot:
    name = "fakespot"
    ID = "fakespot-ai/roberta-base-ai-text-detection-v1"

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(self.ID)
        self.m = AutoModelForSequenceClassification.from_pretrained(self.ID).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, text):
        enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        return torch.softmax(self.m(**enc).logits.float(), -1)[0, 1].item()  # label 1 = AI


ALL = [Binoculars, Desklib, Fakespot]


class Bank:
    """Lazily loads all detectors; unload() frees the GPU (shared host)."""

    def __init__(self):
        self.d = None
        self.lock = threading.Lock()

    def features(self, text):
        with self.lock:
            if self.d is None:
                self.d = [cls() for cls in ALL]
            return {det.name: det(text) for det in self.d}

    def unload(self):
        with self.lock:
            if self.d is not None:
                self.d = None
                import gc
                gc.collect()
                if DEV == "cuda":
                    torch.cuda.empty_cache()

    @property
    def loaded(self):
        return self.d is not None
