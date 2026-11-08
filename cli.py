"""Client: sends files to the service and prints a report, most AI-like paragraphs first."""
import argparse, glob, json, os, sys, urllib.error, urllib.request

URL = os.environ.get("AIDET_URL", "http://127.0.0.1:8790")


def main():
    ap = argparse.ArgumentParser(description="Detect AI-generated text (.tex/.md/.txt) via the ai-detector service")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=8, help="paragraphs to list per file (most AI-like first)")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--sentences", action="store_true", help="also list the most AI-like sentences of flagged paragraphs")
    ap.add_argument("--fail-above", type=float, metavar="PCT",
                    help="exit with status 1 if more than PCT%% of the words are labelled ai (for CI / pre-commit)")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        if p == "-":  # text from stdin
            files.append("-")
            continue
        files += sorted(f for e in ("tex", "md", "txt") for f in glob.glob(os.path.join(p, f"*.{e}"))) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("no .tex/.md/.txt files found")
    tot = {"ai": 0.0, "uncertain": 0.0, "human": 0.0}
    for f in files:
        text = sys.stdin.read() if f == "-" else open(f, encoding="utf-8", errors="replace").read()
        body = json.dumps({"text": text, "min_words": a.min_words,
                           "latex": True if f.endswith(".tex") else None}).encode()
        req = urllib.request.Request(URL + "/api/detect", body, {"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        except urllib.error.HTTPError as e:
            sys.exit(f"{f}: service error {e.code}: {e.read().decode()[:200]}")
        except OSError as e:
            sys.exit(f"service not reachable at {URL} ({e}); start it with: systemctl start ai-detector")
        s = d["share_words"]
        words = sum(p["words"] for p in d["paragraphs"])
        for k in tot:
            tot[k] += s[k] * words
        if a.json:
            print(json.dumps({"file": f, **d}))
            continue
        print(f"\n{os.path.basename(f)}: {d['verdict'].upper()}  (AI {s['ai']:.0%}, uncertain {s['uncertain']:.0%}, "
              f"human {s['human']:.0%} of words; {d['n_paragraphs']} paragraphs)")
        for p in sorted(d["paragraphs"], key=lambda p: -p["score"])[:a.top]:
            if p["label"] != "human":
                print(f"   {p['label']:9s} {p['score']:6.2f}  {p['text'][:100]}...")
                if a.sentences:
                    for x in sorted(p["sentences"], key=lambda x: -x["score"])[:3]:
                        if x["score"] >= d["sentence_threshold_5"]:
                            print(f"      {x['score']:5.1f}  {x['text'][:110]}")
        sd = d.get("structure_distance")
        if sd:
            print(f"   structural distance from human papers: {sd['distance']} ({sd['level']}; human p95 {sd['p95']}, p99 {sd['p99']})")
        traits = [t for t in d.get("structure_traits", []) if abs(t["z"]) >= 1.5][:5]
        if traits:
            print("   structure far from human papers: " + "; ".join(
                f"{t['trait']} {t['value']} (human {t['human_mean']})" for t in traits))
    w = sum(tot.values()) or 1
    if not a.json and len(files) > 1:
        print(f"\nALL FILES: AI {tot['ai']/w:.0%}, uncertain {tot['uncertain']/w:.0%}, human {tot['human']/w:.0%} of words")
    if a.fail_above is not None and tot["ai"] / w * 100 > a.fail_above:
        print(f"FAIL: {tot['ai']/w:.0%} of words labelled ai (limit {a.fail_above:g}%)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
