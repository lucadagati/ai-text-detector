"""Score every paragraph of a LaTeX (or Markdown/plain) file through the running service.

  score_tex.py FILE [--sentences] [--only-flagged]
Prints, per source paragraph (blank-line separated), its first line number, label, score and opening words;
with --sentences, the window score of each sentence (highest first) for flagged paragraphs.
Uses AIDET_URL (default http://127.0.0.1:8790).
"""
import argparse, json, os, re, urllib.request

URL = os.environ.get("AIDET_URL", "http://127.0.0.1:8790")


def detect(text, latex):
    body = json.dumps({"text": text, "latex": latex, "min_words": 40}).encode()
    req = urllib.request.Request(URL + "/api/detect", body, {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def blocks(src):
    """(first line number, text) of blank-line separated blocks, keeping environments whole."""
    out, cur, start, depth = [], [], 1, 0
    for i, line in enumerate(src.splitlines(), 1):
        depth += len(re.findall(r"\\begin\{(?:figure|table|tikzpicture|equation|align|enumerate|itemize|tabularx?)\*?\}", line))
        depth -= len(re.findall(r"\\end\{(?:figure|table|tikzpicture|equation|align|enumerate|itemize|tabularx?)\*?\}", line))
        if not line.strip() and depth <= 0 and cur:
            out.append((start, "\n".join(cur))); cur = []
        elif line.strip():
            if not cur:
                start = i
            cur.append(line)
    if cur:
        out.append((start, "\n".join(cur)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--sentences", action="store_true")
    ap.add_argument("--only-flagged", action="store_true")
    a = ap.parse_args()
    src = open(a.file, encoding="utf-8").read()
    latex = a.file.endswith(".tex")
    n = {"ai": 0, "uncertain": 0, "human": 0}
    for line, b in blocks(src):
        d = detect(b, latex)
        for p in d["paragraphs"]:
            n[p["label"]] += 1
            if a.only_flagged and p["label"] == "human":
                continue
            print(f"L{line:<5d} {p['label']:9s} {p['score']:6.2f}  {p['text'][:90]}")
            if a.sentences and p["label"] != "human":
                thr = d["sentence_threshold"]
                for s in sorted(p["sentences"], key=lambda s: -s["score"])[:4]:
                    mark = "!!" if s["score"] >= thr else ("! " if s["score"] >= d["sentence_threshold_5"] else "  ")
                    print(f"        {mark} {s['score']:5.1f}  {s['text'][:110]}")
    tot = sum(n.values()) or 1
    print(f"\nTOTAL {tot} paragraphs: ai {n['ai']} ({n['ai']/tot:.0%}), uncertain {n['uncertain']} "
          f"({n['uncertain']/tot:.0%}), human {n['human']} ({n['human']/tot:.0%})")


if __name__ == "__main__":
    main()
