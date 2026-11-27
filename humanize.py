"""Humanizer + detector loop: rewrite flagged paragraphs until the detector calls them human.

  humanize.py FILE [-o OUT | --inplace] [--rounds 3] [--candidates 3] [--model sonnet] [--sample SAMPLE.tex]
              [--budget TIER=N] [--max-math 0.3] [--full-rules] [--all] [--dry-run]

For every prose block (blank-line separated, LaTeX/Markdown/plain):
  1. score it with the running service (AIDET_URL);
  2. if it is `ai` or `uncertain` (or --all), generate --candidates rewrites with `claude -p`, using the academic guidelines
     (humanizer/ACADEMIC.md), a compact digest of the humanizer patterns (humanizer/COMPACT.md; the full vendored
     blader/humanizer rules with --full-rules) and the detector feedback (worst sentences, measured structure gaps as instructions);
  3. reject a candidate that breaks the academic guidelines (humanizer/ACADEMIC.md, checked by academic_lint.py: new stock
     formulas, hype, contractions, lost technical terms/acronyms, longer sentences) or changes numbers, \\cite/\\ref/\\label/\\Cref keys, or inline math;
  4. rescore, keep the lowest valid candidate, stop at `human`, after --rounds or after two rounds without gain.
The original is never touched unless --inplace. Pair the output with track_changes.py to get a tracked layer:
  track_changes.py ORIGINAL.tex OUT.tex sections/FILE.tex
"""
import argparse, collections, concurrent.futures as cf, os, re, subprocess, sys, threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_tex import detect, blocks  # noqa: E402
from academic_lint import violations, lint  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACADEMIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanizer", "ACADEMIC.md")
HUMANIZER = os.environ.get("HUMANIZER_SKILL", os.path.join(HERE, "humanizer", "SKILL.md"))


COMPACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humanizer", "COMPACT.md")
FULL_RULES = False


def rules():
    if not FULL_RULES:  # ~1 KB instead of ~32 KB per call: cheaper, and fits a local model's 8k context
        return open(COMPACT, encoding="utf-8").read()
    t = open(HUMANIZER, encoding="utf-8").read()
    return re.sub(r"\A---.*?---\s*", "", t, flags=re.S)  # drop YAML front matter


def worst(d):
    ps = d["paragraphs"]
    if not ps:
        return None
    p = max(ps, key=lambda p: p["score"])
    return p["label"], p["score"], d


def score(text, latex):
    return worst(detect(text, latex))


def invariants(t):
    """Things a rewrite must not change: numbers, reference commands, inline math."""
    nums = collections.Counter(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\\(?:cite|ref|Cref|cref|label|eqref)\w*\{[^}]*\}", "", t)))
    refs = collections.Counter(re.findall(r"\\(?:cite|citep|citet|ref|Cref|cref|eqref|label)\w*\{[^}]*\}", t))
    math = collections.Counter(re.findall(r"\$[^$]+\$", t))
    return nums, refs, math


DIRECT = {  # trait -> (instruction when value is ABOVE human mean, instruction when BELOW)
    "passive %": ("turn passives into active clauses with a named subject (we, the gate, the controller)", "use a passive where the agent is irrelevant"),
    "contains ': ' mid-sentence %": ("remove mid-sentence colons: join with a verb or split into two sentences", "a colon may introduce a definition or list"),
    "contains ';' %": ("replace semicolons with a full stop or a conjunction", "a semicolon may join two closely linked clauses"),
    "ends with ', not X' %": ("delete ', not X' tails", ""),
    "opens with subordinate clause %": ("open fewer sentences with When/If/Because/Although; put the subject first", "open some sentences with a subordinate clause"),
    "opens with prepositional phrase %": ("open fewer sentences with In/On/For/With ...; put the subject first", "open some sentences with a prepositional phrase"),
    "opens with adverb / connective %": ("drop sentence-initial connectives", "open some sentences with a connective that states the logical relation (However, Therefore, In practice, For example)"),
    "opens with subject (noun phrase) %": ("vary openings", "open more sentences directly with their subject"),
    "sentence length, CV": ("", "vary sentence length more: mix a short sentence (under 10 words) with a long one (30+ words) that carries a real relation"),
    "short sentences (<10 words) %": ("merge some short sentences into their neighbours", "add a short sentence for the key result"),
    "long sentences (>35 words) %": ("split sentences over 35 words", "join two related sentences into one longer sentence"),
    "simple sentences (no subordinate/coordinate clause) %": ("combine simple sentences with a subordinate or coordinate clause where the logic is causal or conditional", "keep some sentences simple"),
    "subordinate clauses per sentence": ("remove empty subordinate clauses", "express reasons and conditions as subordinate clauses (because, so that, which)"),
    "coordinated clauses per sentence": ("split long and/but chains", "join some sentences with and/but/so"),
    "noun/verb ratio": ("replace nominalizations with verbs (the evaluation of X -> we evaluate X; the enforcement of -> enforces)", "use more nominal phrases"),
    "nominalizations per word %": ("replace nominalizations with verbs", "use a noun where it is the precise term"),
    "copular 'X is Y' %": ("replace 'X is Y' definitions with a verb that says what X does", "state a definition with 'is' once if needed"),
    "lists of 3+ items %": ("break up enumerations of three or more items; keep only the ones the text needs", "an enumeration is acceptable"),
    "first person in sentence %": ("use fewer 'we'", "state the authors' own choices in the first person ('we measure', 'we choose X because')"),
    "appositives/parentheses per sentence": ("remove parentheses and appositives that do not carry data", "an appositive may carry a definition or a number"),
    "sentence length, mean": ("shorten the average sentence", "lengthen the average sentence by joining related ideas"),
    "rhythm: |Δlength| between consecutive sentences / mean": ("", "alternate sentence length: follow a long sentence with a short one"),
}


def directives(d):
    out = []
    for t in d.get("structure_traits", []):
        if abs(t["z"]) < 1.5 or t["trait"] not in DIRECT:
            continue
        hi, lo = DIRECT[t["trait"]]
        ins = hi if t["z"] > 0 else lo
        if ins:
            out.append(f"- {t['trait']}: {t['value']} vs human papers {t['human_mean']} -> {ins}")
    return "\n".join(out)


def feedback(d):
    out = []
    ss = [s for p in d["paragraphs"] for s in p["sentences"]]
    for s in sorted(ss, key=lambda s: -s["score"])[:3]:
        if s["score"] >= d["sentence_threshold_5"]:
            out.append(f"- most AI-like sentence ({s['score']:.1f}): {s['text'][:220]}")
    dr = directives(d)
    if dr:
        out.append("MEASURED STRUCTURE GAPS (fix these, in this order; they are what the detector reads):\n" + dr)
    return "\n".join(out) or "- no single sentence stands out; the whole paragraph reads as default prose"


def traits():
    # academic guidelines win; the measured gaps in the feedback block are the detector-specific part
    return open(ACADEMIC, encoding="utf-8").read()


def build_prompt(par, d, sample, latex):
    kind = "LaTeX" if latex else "text"
    s = f"""You edit one paragraph of an academic manuscript ({kind}). Apply the humanizer rules below and keep the meaning.

HARD CONSTRAINTS
- Keep every number, every \\cite/\\ref/\\Cref/\\label key and every inline math expression exactly as written.
- Do not add a fact, name, number, claim, inference, recommendation or citation. Do not drop a claim. Length within +-20% of the original.
- Do not insert 'we think/we expect/in our view' statements that state opinions the paragraph does not already hold.
- Do not replace technical terms, acronyms or variable names with synonyms; do not add stock formulas, hype or colloquialisms.
- Keep LaTeX commands and the paragraph's role; return ONE paragraph, no heading, no commentary, no code fence.

{traits()}

Feedback on the current version:
{feedback(d)}
"""
    if sample:
        s += f"\nWRITING SAMPLE (match its sentence length, word choice, openings and transitions):\n{sample}\n"
    s += f"\nHUMANIZER RULES\n{rules()}\n\nPARAGRAPH TO REWRITE\n{par}\n"
    return s


SYSTEM = ("You are a copy editor. Output only the rewritten paragraph, in the same language as the input, "
          "with no preface, notes or code fence. Never add claims that the input does not make.")


def rewrite_ollama(prompt, model):
    # local model through Ollama; num_ctx 8192 so the request shares the loaded instance with other users
    import json, urllib.request
    body = {"model": model, "stream": False, "keep_alive": "10m",
            "options": {"num_ctx": 8192, "temperature": 0.7},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if model.startswith("gpt-oss"):
        body["think"] = "low"
    req = urllib.request.Request(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)["message"]["content"].strip()
    out = re.sub(r"^```\w*\n|\n```$", "", out).strip()
    if not out:
        raise RuntimeError("empty output from ollama")
    return out


def rewrite(prompt, model):
    if model.startswith("ollama:"):
        return rewrite_ollama(prompt, model.split(":", 1)[1])
    # neutral system prompt, no user settings/CLAUDE.md, empty cwd: the editor must not pick up chat-language rules
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text", "--setting-sources", "",
                        "--system-prompt", SYSTEM, prompt],
                       capture_output=True, text=True, timeout=600, cwd="/tmp")
    out = r.stdout.strip()
    out = re.sub(r"^```\w*\n|\n```$", "", out).strip()
    if not out:
        raise RuntimeError("empty output: " + r.stderr[:200])
    return out


_lock = threading.Lock()
CALLS = collections.Counter()
BUDGET = {}


def math_share(t):
    return sum(len(m) for m in re.findall(r"\$[^$]+\$", t)) / max(1, len(t))


def pick_tier(tiers, r, stalled):
    """Cheap tier first; move up one tier per failed round; a tier with exhausted budget is skipped."""
    i = min(r if stalled else 0, len(tiers) - 1)
    while i >= 0:
        m = tiers[i]
        with _lock:
            if BUDGET.get(m) is None or CALLS[m] < BUDGET[m]:
                CALLS[m] += 1
                return m
        i -= 1
    return None


def process(item, a, sample, latex):
    line, par = item
    s0 = score(par, latex)
    if s0 is None:
        return line, par, None, None, "skip"
    if s0[0] != "human" and math_share(par) > a.max_math:
        return line, par, (s0[0], s0[1]), (s0[0], s0[1]), f"skipped (math share {math_share(par):.0%}: detector gain unlikely)"
    label, sc, d = s0
    if label == "human" and not a.all:
        return line, par, (label, sc), (label, sc), "kept"
    best, best_s, cur, cur_d = par, (label, sc), par, d
    inv0 = invariants(par)
    note = "no gain"
    tiers, fails = a.model.split(","), 0
    for r in range(a.rounds):
        if a.dry_run:
            break
        m = pick_tier(tiers, fails, fails > 0)
        if m is None:
            note = "stopped (budget exhausted)"
            break
        prompt = build_prompt(cur, cur_d, sample, latex)
        scored, why = [], collections.Counter()
        for _ in range(a.candidates):
            try:
                with _lock:
                    CALLS[m + " candidates"] += 1
                cand = rewrite(prompt, m)
            except Exception as e:  # noqa
                why["error"] += 1
                continue
            if len(cand.split()) > 1.25 * len(par.split()):
                why["grew"] += 1
                continue
            if invariants(cand) != inv0:
                why["numbers/refs/math"] += 1
                continue
            bad = violations(par, cand)
            if bad:
                why[bad[0].split(":")[0]] += 1
                continue
            sc2 = score(cand, latex)
            if sc2 is not None:
                scored.append((sc2[1], cand, sc2))
        if not scored:
            note = "rejected (" + ", ".join(f"{k} x{v}" for k, v in why.items()) + ")"
            fails += 1
            continue
        _, cand, s = min(scored, key=lambda x: x[0])
        cur, cur_d = cand, s[2]
        if s[1] < best_s[1] - 0.05:
            best, best_s, note, fails = cand, (s[0], s[1]), f"round {r+1} best of {len(scored)}", 0
        else:
            fails += 1
        if s[0] == "human":
            break
        if fails >= 2 and len(tiers) == 1:
            break
    return line, best, (label, sc), best_s, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="sonnet",
                    help="comma list of tiers, cheapest first (claude model or ollama:NAME); a paragraph moves up a tier after a failed round")
    ap.add_argument("--candidates", type=int, default=3, help="candidates generated per round; the best valid one is kept")
    ap.add_argument("--budget", action="append", default=[], metavar="TIER=N", help="max calls for a tier, e.g. opus=15 (repeatable)")
    ap.add_argument("--max-math", type=float, default=0.30, help="skip flagged paragraphs whose inline math exceeds this share of characters")
    ap.add_argument("--full-rules", action="store_true", help="send the whole humanizer SKILL.md (32 KB) instead of the compact digest")
    ap.add_argument("--sample", help="file with your own writing, to match the voice")
    ap.add_argument("--all", action="store_true", help="also rewrite paragraphs already labelled human")
    ap.add_argument("--dry-run", action="store_true", help="score only")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--tracked", metavar="TRACKED_OUT", help="also write a `changes`-package tracked layer over the original (LaTeX)")
    a = ap.parse_args()
    if not (a.out or a.inplace or a.dry_run):
        ap.error("give -o OUT, --inplace or --dry-run")
    global FULL_RULES
    FULL_RULES = a.full_rules
    for b in a.budget:
        k, v = b.rsplit("=", 1)
        BUDGET[k] = int(v)
    src = open(a.file, encoding="utf-8").read()
    latex = a.file.endswith(".tex")
    sample = open(a.sample, encoding="utf-8").read()[:2500] if a.sample else None
    bl = blocks(src)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda b: process(b, a, sample, latex), bl))
    out, n = src, collections.Counter()
    for (line, new, before, after, note), (_, old) in zip(res, bl):
        if before:
            n[(before[0], after[0])] += 1
            print(f"L{line:<5d} {before[0]:9s}{before[1]:6.2f} -> {after[0]:9s}{after[1]:6.2f}  {note}")
        if new != old:
            out = out.replace(old, new, 1)
    ch = sum(v for (b, af), v in n.items() if b != "human")
    ok = sum(v for (b, af), v in n.items() if b != "human" and af == "human")
    print(f"\nflagged before: {ch}, human after: {ok}; words {len(src.split())} -> {len(out.split())}")
    if a.dry_run:
        return
    dst = a.file if a.inplace else a.out
    if a.inplace and a.tracked:
        ap.error("--tracked needs -o (the original must stay untouched)")
    open(dst, "w", encoding="utf-8").write(out)
    print("written", dst)
    if a.tracked:
        subprocess.run([sys.executable, os.path.join(HERE, "track_changes.py"), a.file, dst, a.tracked], check=True)
        print("tracked layer written", a.tracked)
    print("calls: " + (", ".join(f"{k}={v}" for k, v in CALLS.items()) or "none"))
    a0, a1 = lint(src), lint(out)
    print("guideline lint (before -> after): " + ", ".join(f"{k} {a0[k]} -> {a1[k]}" for k in ("formulas", "hype", "contractions", "repeat_openings")))
    print("Result is optimized against this detector only; check structure distance with aidetect.")


if __name__ == "__main__":
    main()
