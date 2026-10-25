"""Client: sends files to the service and prints a report, most AI-like paragraphs first."""
import argparse, glob, json, os, sys, urllib.request

URL = os.environ.get("AIDET_URL", "http://127.0.0.1:8790")


def main():
    ap = argparse.ArgumentParser(description="Detect AI-generated text (.tex/.md/.txt) via the ai-detector service")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=8, help="paragraphs to list per file (most AI-like first)")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(f for e in ("tex", "md", "txt") for f in glob.glob(os.path.join(p, f"*.{e}"))) if os.path.isdir(p) else [p]
    tot = {"ai": 0.0, "uncertain": 0.0, "human": 0.0}
    for f in files:
        body = json.dumps({"text": open(f, encoding="utf-8", errors="replace").read(), "min_words": a.min_words,
                           "latex": True if f.endswith(".tex") else None}).encode()
        req = urllib.request.Request(URL + "/api/detect", body, {"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        except OSError as e:
            sys.exit(f"service not reachable at {URL} ({e}); start it with: systemctl start ai-detector")
        if a.json:
            print(json.dumps({"file": f, **d}))
            continue
        s = d["share_words"]
        words = sum(p["words"] for p in d["paragraphs"])
        for k in tot:
            tot[k] += s[k] * words
        print(f"\n{os.path.basename(f)}: {d['verdict'].upper()}  (AI {s['ai']:.0%}, uncertain {s['uncertain']:.0%}, "
              f"human {s['human']:.0%} of words; {d['n_paragraphs']} paragraphs)")
        for p in sorted(d["paragraphs"], key=lambda p: -p["score"])[:a.top]:
            if p["label"] != "human":
                print(f"   {p['label']:9s} {p['score']:6.2f}  {p['text'][:100]}...")
        sd = d.get("structure_distance")
        if sd:
            print(f"   structural distance from human papers: {sd['distance']} ({sd['level']}; human p95 {sd['p95']}, p99 {sd['p99']})")
        traits = [t for t in d.get("structure_traits", []) if abs(t["z"]) >= 1.5][:5]
        if traits:
            print("   structure far from human papers: " + "; ".join(
                f"{t['trait']} {t['value']} (human {t['human_mean']})" for t in traits))
    if not a.json and len(files) > 1:
        w = sum(tot.values()) or 1
        print(f"\nALL FILES: AI {tot['ai']/w:.0%}, uncertain {tot['uncertain']/w:.0%}, human {tot['human']/w:.0%} of words")


if __name__ == "__main__":
    main()
