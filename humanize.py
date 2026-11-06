"""Humanizer + detector loop: rewrite flagged paragraphs until the detector calls them human.

  humanize.py FILE [-o OUT | --inplace] [--rounds 4] [--model sonnet] [--sample SAMPLE.tex]
              [--all] [--dry-run]

For every prose block (blank-line separated, LaTeX/Markdown/plain):
  1. score it with the running service (AIDET_URL);
  2. if it is `ai` or `uncertain` (or --all), rewrite it with `claude -p`, using the humanizer rules
     (humanizer/SKILL.md, vendored from blader/humanizer, MIT) plus the detector feedback (worst sentences,
     structure traits farthest from human papers);
  3. reject a candidate that changes numbers, \\cite/\\ref/\\label/\\Cref keys, or inline math;
  4. rescore, keep the best candidate, stop at `human` or after --rounds.
The original is never touched unless --inplace. Pair the output with track_changes.py to get a tracked layer:
  track_changes.py ORIGINAL.tex OUT.tex sections/FILE.tex
"""
import argparse, collections, concurrent.futures as cf, os, re, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_tex import detect, blocks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HUMANIZER = os.environ.get("HUMANIZER_SKILL", os.path.join(HERE, "humanizer", "SKILL.md"))


def rules():
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


def feedback(d):
    out = []
    ss = [s for p in d["paragraphs"] for s in p["sentences"]]
    for s in sorted(ss, key=lambda s: -s["score"])[:3]:
        if s["score"] >= d["sentence_threshold_5"]:
            out.append(f"- most AI-like sentence ({s['score']:.1f}): {s['text'][:220]}")
    return "\n".join(out) or "- no single sentence stands out; the whole paragraph reads as default prose"


TRAITS = """Detector findings on AI prose in this field (lower the score by doing the opposite):
- long connected sentences with subordinate clauses, first person ("we", "in our view"), "e.g."/"i.e.", concrete
  examples from the paper's own system, explicit motivations, occasional passive, connective openings
  ("However", "In practice", "For example") lower it;
- short punchy declaratives, ", not X" tails, colon reveals, semicolon chains, generic definitional sentences raise it."""


def build_prompt(par, d, sample, latex):
    kind = "LaTeX" if latex else "text"
    s = f"""You edit one paragraph of an academic manuscript ({kind}). Apply the humanizer rules below and keep the meaning.

HARD CONSTRAINTS
- Keep every number, every \\cite/\\ref/\\Cref/\\label key and every inline math expression exactly as written.
- Do not add a fact, name, number, claim, inference, recommendation or citation. Do not drop a claim. Length within +-20% of the original.
- Do not insert 'we think/we expect/in our view' statements that state opinions the paragraph does not already hold.
- Keep LaTeX commands and the paragraph's role; return ONE paragraph, no heading, no commentary, no code fence.

{TRAITS}

Feedback on the current version:
{feedback(d)}
"""
    if sample:
        s += f"\nWRITING SAMPLE (match its sentence length, word choice, openings and transitions):\n{sample}\n"
    s += f"\nHUMANIZER RULES\n{rules()}\n\nPARAGRAPH TO REWRITE\n{par}\n"
    return s


SYSTEM = ("You are a copy editor. Output only the rewritten paragraph, in the same language as the input, "
          "with no preface, notes or code fence. Never add claims that the input does not make.")


def rewrite(prompt, model):
    # neutral system prompt, no user settings/CLAUDE.md, empty cwd: the editor must not pick up chat-language rules
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text", "--setting-sources", "",
                        "--system-prompt", SYSTEM, prompt],
                       capture_output=True, text=True, timeout=600, cwd="/tmp")
    out = r.stdout.strip()
    out = re.sub(r"^```\w*\n|\n```$", "", out).strip()
    if not out:
        raise RuntimeError("empty output: " + r.stderr[:200])
    return out


def process(item, a, sample, latex):
    line, par = item
    s0 = score(par, latex)
    if s0 is None:
        return line, par, None, None, "skip"
    label, sc, d = s0
    if label == "human" and not a.all:
        return line, par, (label, sc), (label, sc), "kept"
    best, best_s, cur, cur_d = par, (label, sc), par, d
    inv0 = invariants(par)
    note = "no gain"
    for r in range(a.rounds):
        if a.dry_run:
            break
        try:
            cand = rewrite(build_prompt(cur, cur_d, sample, latex), a.model)
        except Exception as e:  # noqa
            note = f"error: {e}"
            continue
        if len(cand.split()) > 1.25 * len(par.split()):
            note = "rejected (grew >25%: likely added claims)"
            continue
        if invariants(cand) != inv0:
            note = "rejected (numbers/refs/math changed)"
            continue
        s = score(cand, latex)
        if s is None:
            continue
        cur, cur_d = cand, s[2]
        if s[1] < best_s[1]:
            best, best_s, note = cand, (s[0], s[1]), f"round {r+1}"
        if s[0] == "human":
            break
    return line, best, (label, sc), best_s, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("-o", "--out")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--sample", help="file with your own writing, to match the voice")
    ap.add_argument("--all", action="store_true", help="also rewrite paragraphs already labelled human")
    ap.add_argument("--dry-run", action="store_true", help="score only")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--tracked", metavar="TRACKED_OUT", help="also write a `changes`-package tracked layer over the original (LaTeX)")
    a = ap.parse_args()
    if not (a.out or a.inplace or a.dry_run):
        ap.error("give -o OUT, --inplace or --dry-run")
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
    print("Result is optimized against this detector only; check structure distance with aidetect.")


if __name__ == "__main__":
    main()
