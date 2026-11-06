"""Score every benchmark paragraph with every detector (one detector in memory at a time).
Cache: bench/data/features.jsonl  {id, <detector>: value}"""
import gc, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.detectors import Binoculars, Desklib, Fakespot  # noqa: E402
from aidet.textprep import latex_to_text  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data")
DETS = [("binoculars", lambda: Binoculars("falcon")), ("binoculars_qwen", lambda: Binoculars("qwen")),
        ("desklib", Desklib), ("fakespot", Fakespot)]


def norm(text):
    """Same pipeline for both classes, so no LaTeX/Markdown artifact can separate them."""
    return " ".join(latex_to_text(text.replace("**", "")).split())


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None  # e.g. desklib,fakespot
    rows = [json.loads(l) for f in ("human.jsonl", "human_abs.jsonl", "ai.jsonl") if os.path.exists(os.path.join(D, f))
            for l in open(os.path.join(D, f))]
    for r in rows:
        r["text"] = norm(r["text"])
    path = os.path.join(D, "features.jsonl")
    feats = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            feats[r["id"]] = r
    for name, make in DETS:
        if only and name not in only:
            continue
        todo = [r for r in rows if name not in feats.get(r["id"], {})]
        if not todo:
            continue
        print(name, len(todo), flush=True)
        det = make()
        for i, r in enumerate(todo):
            feats.setdefault(r["id"], {"id": r["id"]})[name] = det(r["text"])
            if i % 100 == 0:
                print(" ", i, flush=True)
        del det
        gc.collect()
        torch.cuda.empty_cache()
        with open(path, "w") as f:
            for v in feats.values():
                f.write(json.dumps(v) + "\n")


if __name__ == "__main__":
    main()
