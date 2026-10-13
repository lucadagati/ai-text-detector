"""Human reference set: body paragraphs of pre-LLM arXiv papers, from their LaTeX sources.

  collect_human.py [--per-cat N] [--per-paper K] [--years 2015-2019]
Appends to bench/data/human.jsonl and skips papers already collected, so it can be rerun to grow the set.
Each line: {id, paper, cat, title, abstract, text}
"""
import argparse, gzip, io, json, os, random, re, sys, tarfile, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aidet.textprep import latex_to_text  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "data", "human.jsonl")
CATS = ["cs.DC", "cs.RO", "cs.SE", "eess.SY", "cs.NI", "cs.CR", "cs.LG", "cs.AI", "cs.CV", "cs.CL",
        "cs.HC", "cs.DB", "cs.IT", "cs.MA", "cs.PF", "cs.AR", "cs.ET", "cs.CY", "eess.SP", "math.OC"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-detector-bench/1.1 (research)"})
    return urllib.request.urlopen(req, timeout=90).read()


def papers(cat, y0, y1, start):
    q = f"cat:{cat}+AND+submittedDate:[{y0}01010000+TO+{y1}12312359]"
    xml = get(f"https://export.arxiv.org/api/query?search_query={q}&start={start}&max_results=100").decode()
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        aid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
        title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
        abs_ = " ".join(re.search(r"<summary>(.*?)</summary>", e, re.S).group(1).split())
        out.append((aid, title, abs_))
    return out


def tex_of(aid):
    req = urllib.request.Request(f"https://arxiv.org/e-print/{aid}", method="HEAD",
                                 headers={"User-Agent": "ai-detector-bench/1.1 (research)"})
    size = int(urllib.request.urlopen(req, timeout=30).headers.get("Content-Length") or 0)
    if size > 15_000_000:  # mostly figures; skip to keep the crawl fast
        raise ValueError(f"source too large ({size >> 20} MB)")
    raw = get(f"https://arxiv.org/e-print/{aid}")
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw))
        return "\n\n".join(tf.extractfile(m).read().decode("utf-8", "replace") for m in tf.getmembers()
                           if m.isfile() and m.name.endswith(".tex"))
    except tarfile.ReadError:
        try:
            return gzip.decompress(raw).decode("utf-8", "replace")
        except OSError:
            return raw.decode("utf-8", "replace")


def body_paragraphs(tex):
    m = re.search(r"\\begin\{document\}(.*?)(\\bibliography|\\begin\{thebibliography\}|\\end\{document\})", tex, re.S)
    tex = m.group(1) if m else tex
    tex = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", tex, flags=re.S)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", latex_to_text(tex))]
    good = []
    for p in paras:
        w = p.split()
        if not 60 <= len(w) <= 260:
            continue
        if sum(c.isalpha() for c in p) / len(p) < 0.75 or p.count("x") / len(w) > 0.15:
            continue  # math-heavy or garbage
        if re.search(r"(?i)\b(chatgpt|gpt-4|large language model)\b", p):
            continue  # guard against mislabeled dates
        good.append(p)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=15)
    ap.add_argument("--per-paper", type=int, default=8)
    ap.add_argument("--years", default="2015-2019")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    y0, y1 = a.years.split("-")
    random.seed(a.seed)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen = set()
    if os.path.exists(OUT):
        seen = {json.loads(l)["paper"] for l in open(OUT)}
    n = 0
    with open(OUT, "a") as f:
        for cat in CATS:
            got = 0
            try:
                cand = papers(cat, y0, y1, random.randint(0, 400))
            except Exception as e:  # noqa: BLE001
                print("query failed", cat, e, flush=True)
                continue
            random.shuffle(cand)
            for aid, title, abs_ in cand:
                if got >= a.per_cat:
                    break
                if aid in seen:  # also cross-listed papers
                    continue
                seen.add(aid)
                time.sleep(3)
                try:
                    paras = body_paragraphs(tex_of(aid))
                except Exception as e:  # noqa: BLE001
                    print("skip", aid, e, flush=True)
                    continue
                if len(paras) < 4:
                    continue
                random.shuffle(paras)
                for i, p in enumerate(paras[:a.per_paper]):
                    f.write(json.dumps({"id": f"{aid}#{i}", "paper": aid, "cat": cat, "title": title,
                                        "abstract": abs_, "text": p}) + "\n")
                    n += 1
                f.flush()
                got += 1
                print(cat, aid, len(paras), flush=True)
    print("new human paragraphs:", n)


if __name__ == "__main__":
    main()
