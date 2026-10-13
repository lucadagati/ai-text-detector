"""AI set, matched to the human set paper by paper.

For each human paragraph not yet used as a seed, one (generator, mode) pair writes one AI paragraph:
  free      a new paragraph for the same paper (title + abstract given), similar length
  human     the same, told to sound human and avoid typical LLM phrasing (adversarial)
  polish    a Q1-editor rewrite of the human paragraph (AI-polished human text)
  revise    a terse, assertive "Reviewer 2" revision of the human paragraph that avoids LLM clichés
            (the way manuscripts are revised with an assistant in practice)
Claude runs through `claude -p`, several paragraphs per call (JSON array) to save calls;
local models run through Ollama. Appends to bench/data/ai.jsonl and resumes.
(generate_ai_v1.py produced the first 476 paragraphs: 3 seeds per paper, all three first modes.)
"""
import argparse, json, os, random, re, subprocess, threading, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "data")
OUT = os.path.join(D, "ai.jsonl")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# share of seeds per generator (weights)
GENS = {"claude:sonnet": 30, "claude:opus": 10, "claude:haiku": 8,
        "ollama:gpt-oss:20b": 9, "ollama:mistral-small:24b": 9, "ollama:qwen2.5:14b-instruct": 9,
        "ollama:gemma3:12b": 9, "ollama:llama3.1:8b": 8, "ollama:phi4:14b": 8}
MODES = ["free", "human", "polish", "revise", "results", "multipass"]   # abstracts use the first four
BATCH = 6
lock = threading.Lock()

TASK = {
    "free": "Write one paragraph (about {n} words) that could appear in the body of this research paper.\n"
            "Title: {title}\nAbstract: {abstract}",
    "human": "Write one paragraph (about {n} words) that could appear in the body of this research paper. It must read "
             "as written by an experienced human researcher: vary sentence length, avoid stock LLM phrasing "
             "(e.g. 'Moreover', 'crucial', 'delve', 'not X but Y'), be concrete.\nTitle: {title}\nAbstract: {abstract}",
    "polish": "You are a senior editor of a Q1 IEEE journal. Rewrite this paragraph so that it is clear, assertive and "
              "polished, keeping its meaning, citations and length.\nParagraph: {text}",
    "revise": "Act as a strict 'Reviewer 2' and senior editor. Revise this paragraph for a Q1 journal: remove verbosity "
              "and typical LLM patterns ('Moreover', 'It is important to note', banal summaries), make it assertive "
              "and natural, alternate short punchy claims with longer analytical sentences, keep every technical "
              "fact.\nParagraph: {text}",
    "results": "Write one paragraph (about {n} words) of the Results/Evaluation section of this research paper, "
               "reporting concrete quantitative findings (numbers, percentages, latencies, comparisons with baselines, "
               "statistical tests).\nTitle: {title}\nAbstract: {abstract}",
    "multipass": "Revise this paragraph three times in succession, as three different strict editors of a Q1 journal: "
                 "first tighten it, then make it assertive and remove any LLM-sounding phrasing, then compress it into "
                 "dense, precise technical prose. Keep every technical fact. Output only the final version.\nParagraph: {text}",
}
TASK_ABS = {
    "free": "Write the abstract (about {n} words) of a research paper titled: {title}",
    "human": "Write the abstract (about {n} words) of a research paper titled: {title}\nIt must read as written by an "
             "experienced human researcher: avoid stock LLM phrasing (e.g. 'Moreover', 'crucial', 'delve', "
             "'not X but Y'), be concrete.",
    "polish": TASK["polish"].replace("this paragraph", "this abstract").replace("Paragraph:", "Abstract:"),
    "revise": TASK["revise"].replace("this paragraph", "this abstract").replace("Paragraph:", "Abstract:"),
}
FMT_ONE = "\n\nOutput only the paragraph, in English, plain prose (LaTeX like \\cite{{key}} allowed), no heading or preamble."
FMT_MANY = ("Below are {k} independent tasks. Do each one separately. Output ONLY a JSON array of {k} strings, "
            "the i-th string being the text for task i, in English, plain prose (LaTeX allowed), "
            "no headings, no comments.\n\n{tasks}")


def ollama(model, prompt):
    think = "low" if model.startswith("gpt-oss") else False  # gpt-oss ignores think=false; "low" keeps it fast
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": think,
                       "options": {"num_ctx": 8192, "temperature": 0.8}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]


def claude(model, prompt):
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text", prompt],
                       capture_output=True, text=True, timeout=900, cwd="/tmp")
    return r.stdout


def clean(t):
    t = t.strip().strip('"')
    t = re.sub(r"^(Here('s| is)[^\n]*:|Sure[^\n]*:|Rewritten[^\n]*:|Revised[^\n]*:)\s*", "", t, flags=re.I)
    paras = [p for p in re.split(r"\n\s*\n", t) if len(p.split()) > 30]
    return max(paras, key=len) if paras else t


def parse_many(out, k):
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(arr, list) and len(arr) == k and all(isinstance(x, str) for x in arr):
        return [clean(x) for x in arr]
    return None


def write(job, text):
    if len(text.split()) < 40:
        return
    with lock, open(OUT, "a") as f:
        f.write(json.dumps({"id": f"{job['hid']}|{job['gen']}|{job['mode']}", "paper": job["paper"], "kind": job["kind"],
                            "gen": job["gen"], "mode": job["mode"], "text": text}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max new seeds (0 = all unused human paragraphs)")
    ap.add_argument("--only", default="", help="comma-separated generator prefixes to run (e.g. claude)")
    a = ap.parse_args()
    human = [json.loads(l) for name in ("human.jsonl", "human_abs.jsonl") if os.path.exists(os.path.join(D, name))
             for l in open(os.path.join(D, name))]
    used = set()
    if os.path.exists(OUT):
        used = {json.loads(l)["id"].split("|")[0] for l in open(OUT)}
    names, w = list(GENS), list(GENS.values())
    jobs = []
    for h in human:
        if h["id"] in used:
            continue
        rnd = random.Random(zlib.crc32(h["id"].encode()))  # stable assignment per seed
        gen = rnd.choices(names, w)[0]
        mode = rnd.choice(MODES[:4] if h.get("kind") == "abstract" else MODES)
        jobs.append({"hid": h["id"], "paper": h["paper"], "gen": gen, "mode": mode, "kind": h.get("kind", "body"),
                     "prompt": (TASK_ABS if h.get("kind") == "abstract" else TASK)[mode].format(n=len(h["text"].split()), title=h["title"],
                                                 abstract=h["abstract"], text=h["text"])})
    if a.only:
        jobs = [j for j in jobs if any(j["gen"].startswith(p) for p in a.only.split(","))]
    if a.limit:
        jobs = jobs[:a.limit]
    print(len(jobs), "jobs", flush=True)

    def run_claude(batch):
        model = batch[0]["gen"].split(":", 1)[1]
        tasks = "\n\n".join(f"### Task {i + 1}\n{j['prompt']}" for i, j in enumerate(batch))
        try:
            raw = claude(model, FMT_MANY.format(k=len(batch), tasks=tasks))
            outs = parse_many(raw, len(batch))
        except Exception as e:  # noqa: BLE001
            print("fail", model, e, flush=True)
            return
        if outs is None:
            print("unparsable batch", model, repr(raw[:200]), flush=True)
            return
        for j, t in zip(batch, outs):
            write(j, t)
        print("ok claude", model, len(batch), flush=True)

    def run_local(j):
        try:
            write(j, clean(ollama(j["gen"].split(":", 1)[1], j["prompt"] + FMT_ONE)))
            print("ok", j["gen"], j["mode"], flush=True)
        except Exception as e:  # noqa: BLE001
            print("fail", j["gen"], e, flush=True)

    cl = [j for j in jobs if j["gen"].startswith("claude")]
    batches = []
    for g in sorted({j["gen"] for j in cl}):
        js = [j for j in cl if j["gen"] == g]
        batches += [js[i:i + BATCH] for i in range(0, len(js), BATCH)]
    local = sorted((j for j in jobs if not j["gen"].startswith("claude")), key=lambda j: j["gen"])
    with ThreadPoolExecutor(4) as ex1, ThreadPoolExecutor(4) as ex2:
        fs = [ex1.submit(run_claude, b) for b in batches] + [ex2.submit(run_local, j) for j in local]
        for f in fs:
            f.result()


if __name__ == "__main__":
    main()
