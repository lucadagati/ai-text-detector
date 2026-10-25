"""AI set, matched to the human set paper by paper.
For sampled human paragraphs, each generator writes
  free      a new paragraph for the same paper (title + abstract given), similar length
  human     the same, but told to sound human and avoid typical LLM phrasing (adversarial)
  polish    a Q1-editor rewrite of the human paragraph (AI-polished human text)
Generators: Claude (claude -p: sonnet, opus, haiku) and local models via Ollama.
Output: bench/data/ai.jsonl  {id, paper, gen, mode, text}"""
import json, os, random, re, subprocess, sys, threading, zlib
from concurrent.futures import ThreadPoolExecutor
import urllib.request

D = os.path.join(os.path.dirname(__file__), "data")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
GENS = ["claude:sonnet", "claude:opus", "claude:haiku", "ollama:gpt-oss:20b", "ollama:mistral-small:24b",
        "ollama:llama3.1:8b", "ollama:qwen2.5:14b-instruct", "ollama:gemma3:12b"]
PER_PAPER = 3          # human paragraphs used per paper
random.seed(1)
lock = threading.Lock()

P_FREE = ("Write one paragraph (about {n} words) that could appear in the body of the research paper below. "
          "Plain academic prose in English, LaTeX allowed (e.g. \\cite{{key}}), no heading, no preamble: "
          "output only the paragraph.\n\nTitle: {title}\nAbstract: {abstract}")
P_HUMAN = P_FREE + ("\n\nIt must read as written by an experienced human researcher: vary sentence length, "
                    "avoid stock LLM phrasing (e.g. 'Moreover', 'crucial', 'delve', 'not X but Y'), "
                    "be concrete and slightly informal where a human would be.")
P_POLISH = ("You are a senior editor of a Q1 IEEE journal. Rewrite the paragraph below so that it is clear, "
            "assertive and polished, keeping its meaning, citations and length. Output only the rewritten "
            "paragraph.\n\n{text}")


def ask(gen, prompt):
    kind, model = gen.split(":", 1)
    if kind == "claude":
        r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text", prompt],
                           capture_output=True, text=True, timeout=300, cwd="/tmp")
        return r.stdout
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"num_ctx": 8192, "temperature": 0.8}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["response"]


def clean(t):
    t = t.strip().strip('"')
    t = re.sub(r"^(Here('s| is)[^\n]*:|Sure[^\n]*:|Rewritten[^\n]*:)\s*", "", t, flags=re.I)
    paras = [p for p in re.split(r"\n\s*\n", t) if len(p.split()) > 30]
    return max(paras, key=len) if paras else t


def main():
    human = [json.loads(l) for l in open(os.path.join(D, "human.jsonl"))]
    byp = {}
    for h in human:
        byp.setdefault(h["paper"], []).append(h)
    out_path = os.path.join(D, "ai.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["id"].split("|")[0], r["mode"]) for r in map(json.loads, open(out_path))}
    jobs = []
    for paper, hs in byp.items():
        for k, h in enumerate(random.sample(hs, min(PER_PAPER, len(hs)))):
            gen = GENS[(zlib.crc32(paper.encode()) + k) % len(GENS)]
            n = len(h["text"].split())
            for mode, tpl in (("free", P_FREE), ("human", P_HUMAN), ("polish", P_POLISH)):
                jid = f"{h['id']}|{gen}|{mode}"
                if (h["id"], mode) not in done:
                    jobs.append((jid, paper, gen, mode, tpl.format(n=n, title=h["title"], abstract=h["abstract"], text=h["text"])))
    print(len(jobs), "jobs", flush=True)

    def run(job):
        jid, paper, gen, mode, prompt = job
        try:
            txt = clean(ask(gen, prompt))
        except Exception as e:  # noqa: BLE001
            print("fail", jid, e, flush=True)
            return
        if len(txt.split()) < 40:
            return
        with lock, open(out_path, "a") as f:
            f.write(json.dumps({"id": jid, "paper": paper, "gen": gen, "mode": mode, "text": txt}) + "\n")
            print("ok", jid, flush=True)

    claude_jobs = [j for j in jobs if j[2].startswith("claude")]
    local_jobs = sorted((j for j in jobs if not j[2].startswith("claude")), key=lambda j: j[2])
    with ThreadPoolExecutor(6) as ex1, ThreadPoolExecutor(4) as ex2:
        f1 = [ex1.submit(run, j) for j in claude_jobs]
        f2 = [ex2.submit(run, j) for j in local_jobs]
        for f in f1 + f2:
            f.result()


if __name__ == "__main__":
    main()
