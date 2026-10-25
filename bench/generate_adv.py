"""Adversarial AI paragraphs: detector-guided rewrites that try to pass the detector (mode `adv`).

Seeds are existing AI paragraphs (bench/data/ai.jsonl). Each seed is rewritten by an LLM given two human
paragraphs of the training papers as style reference and the rules that lower the detector score; the rewrite is
scored by the running service (AIDET_URL) and rewritten again (up to ROUNDS) until its label is `human`.
The last rewrite is kept whether or not it evaded (`evaded`, `rounds`, `score` fields). Appends to
bench/data/ai_adv.jsonl and resumes. Tests held out: papers keep the split of their seed.
  generate_adv.py [--limit N] [--url URL] [--tag adv1]
"""
import argparse, json, os, random, re, subprocess, threading, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "data")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GENS = {"claude:sonnet": 20, "claude:opus": 7, "claude:haiku": 7, "ollama:gpt-oss:20b": 10, "ollama:gemma3:12b": 8,
        "ollama:mistral-small:24b": 8, "ollama:qwen2.5:14b-instruct": 8, "ollama:phi4:14b": 6}
ROUNDS = int(os.environ.get("ADV_ROUNDS", 3))
lock = threading.Lock()

RULES = ("Rewrite the paragraph below so that it reads as written by a human researcher, like the two reference paragraphs.\n"
         "Keep every number, citation (\\cite{{..}}), macro and technical fact; keep the length within 15%.\n"
         "Do: use long connected sentences with subordinate clauses; mix in a few short ones; use first person "
         "('we', 'our') where a paper would; give concrete examples from the setting; state motivations ('because', 'so that'); "
         "use some passive voice; open some sentences with 'However', 'In practice', 'For example', 'Here'; use 'e.g.' / 'i.e.'.\n"
         "Don't: use semicolons, colon reveals, ', not X' endings, em-dashes, tidy lists of three, rhetorical contrasts, "
         "stock words ('crucial', 'notably', 'moreover', 'delve', 'robust'), or a summarizing last sentence.")
AGAIN = "\nA detector still flagged your previous attempt as AI-written. Change the sentence structure more: vary rhythm, be plainer and less polished."


def ollama(model, prompt):
    think = "low" if model.startswith("gpt-oss") else False
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": think,
                       "options": {"num_ctx": 8192, "temperature": 0.9}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]


def claude(model, prompt):
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text", prompt],
                       capture_output=True, text=True, timeout=900, cwd="/tmp")
    if not r.stdout.strip():
        raise RuntimeError("empty claude output: " + r.stderr[:200])
    return r.stdout


def clean(t):
    t = t.strip().strip('"')
    t = re.sub(r"^(Here('s| is)[^\n]*:|Sure[^\n]*:|Rewritten[^\n]*:|Revised[^\n]*:)\s*", "", t, flags=re.I)
    paras = [p for p in re.split(r"\n\s*\n", t) if len(p.split()) > 30]
    return " ".join((max(paras, key=len) if paras else t).split())


FEEDBACK = os.environ.get("ADV_FEEDBACK") == "1"
EDIT = ("Edit the paragraph below so that a detector no longer flags it. Change ONLY the flagged sentences listed (rewrite them "
        "completely with different structure, vocabulary and rhythm, as a human researcher would write them); leave the other "
        "sentences as they are. Keep every number, citation, macro and technical fact.\n"
        "Flagged sentences (most AI-like first, with the detector's score; above 1 is flagged):\n{flags}\n")


def score_full(url, text):
    body = json.dumps({"text": text, "latex": False, "min_words": 20}).encode()
    req = urllib.request.Request(url + "/api/detect", body, {"Content-Type": "application/json"})
    p = json.loads(urllib.request.urlopen(req, timeout=300).read())["paragraphs"]
    return p[0] if p else None


def score(url, text):
    body = json.dumps({"text": text, "latex": False, "min_words": 20}).encode()
    req = urllib.request.Request(url + "/api/detect", body, {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    p = d["paragraphs"]
    return (p[0]["score"], p[0]["label"]) if p else (None, "skipped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--url", default=os.environ.get("AIDET_URL", "http://127.0.0.1:8790"))
    ap.add_argument("--tag", default="adv")
    ap.add_argument("--out", default=os.path.join(D, "ai_adv.jsonl"))
    a = ap.parse_args()
    split = json.load(open(os.path.join(D, "split.json")))
    humans = [json.loads(l) for l in open(os.path.join(D, "human.jsonl"))]
    ref = [h for h in humans if split.get(h["paper"]) == "train" and 60 < len(h["text"].split()) < 200]
    seeds = [json.loads(l) for l in open(os.path.join(D, "ai.jsonl"))]
    if os.environ.get("ADV_SEEDS_FROM"):  # also rewrite earlier adversarial texts (all of them, evaded or not)
        seeds += [json.loads(l) for l in open(os.environ["ADV_SEEDS_FROM"])]
    seeds = [s for s in seeds if s.get("kind", "body") == "body" and 50 <= len(s["text"].split()) <= 260]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["seed"] for l in open(a.out)}
    names, w = list(GENS), list(GENS.values())
    jobs = []
    for s in seeds:
        if s["id"] in done:
            continue
        rnd = random.Random(zlib.crc32((a.tag + s["id"]).encode()))
        jobs.append({**s, "rgen": rnd.choices(names, w)[0], "ref": rnd.sample(ref, 2)})
    random.Random(1).shuffle(jobs)
    if a.limit:
        jobs = jobs[:a.limit]
    print(len(jobs), "seeds", flush=True)

    def run(j):
        kind, model = j["rgen"].split(":", 1)
        gen = (claude if kind == "claude" else ollama)
        text = j["text"]
        fl, sc, lab, rd = [], None, None, 0
        try:
            for rd in range(1, ROUNDS + 1):
                refs = "\n\n".join(f"Reference {i + 1}: {h['text']}" for i, h in enumerate(j["ref"]))
                if FEEDBACK and rd > 1 and fl:
                    p = EDIT.format(flags="\n".join(f"- ({v:.1f}) {t}" for t, v in fl)) + f"\nParagraph:\n{text}\n\nOutput only the edited paragraph, in English, plain prose, no preamble."
                else:
                  p = f"{RULES}{AGAIN if rd > 1 else ''}\n\n{refs}\n\nParagraph to rewrite:\n{text}\n\nOutput only the rewritten paragraph, in English, plain prose, no preamble."
                text = clean(gen(model, p))
                if len(text.split()) < 40:
                    return
                if FEEDBACK:
                    pr = score_full(a.url, text)
                    if pr is None:
                        lab = "skipped"
                        break
                    sc, lab = pr["score"], pr["label"]
                    fl = sorted(((x["text"], x["score"]) for x in pr["sentences"]), key=lambda x: -x[1])[:4]
                else:
                    sc, lab = score(a.url, text)
                if lab in ("human", "skipped"):
                    break
        except Exception as e:  # noqa: BLE001
            print("fail", j["rgen"], repr(e)[:200], flush=True)
            return
        if lab in (None, "skipped"):
            return
        with lock, open(a.out, "a") as f:
            f.write(json.dumps({"id": f"{j['id']}|{j['rgen']}|{a.tag}", "seed": j["id"], "paper": j["paper"], "kind": "body",
                                "gen": j["rgen"], "mode": a.tag, "text": text, "evaded": lab == "human", "rounds": rd,
                                "score": sc}) + "\n")
        print("ok", j["rgen"], rd, lab, flush=True)

    with ThreadPoolExecutor(10) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
