"""Human abstracts (pre-2020 arXiv, plain text from the API) appended to bench/data/human.jsonl with kind=abstract.

  collect_abstracts.py [--per-cat 80]
Abstracts in bench/data/abstracts_pre2020.jsonl (the held-out check of sanity.py) are excluded.
"""
import argparse, json, os, random, re, time, urllib.request

D = os.path.join(os.path.dirname(__file__), "data")
CATS = ["cs.DC", "cs.RO", "cs.SE", "eess.SY", "cs.NI", "cs.CR", "cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.HC", "cs.DB",
        "cs.IT", "cs.MA", "cs.PF", "cs.AR", "cs.ET", "cs.CY", "eess.SP", "math.OC", "cs.DS", "cs.GT", "cs.IR", "cs.SI"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=80)
    a = ap.parse_args()
    held = {json.loads(l)["text"] for l in open(os.path.join(D, "abstracts_pre2020.jsonl"))}
    have = {json.loads(l)["paper"] for l in open(os.path.join(D, "human.jsonl"))}
    rnd = random.Random(5)
    n = 0
    with open(os.path.join(D, "human_abs.jsonl"), "a") as f:  # merged into human.jsonl once the crawler stops
        for cat in CATS:
            q = f"cat:{cat}+AND+submittedDate:[201501010000+TO+201912312359]"
            start = rnd.choice([0, 1000, 2000, 3000])  # sanity.py's held-out set uses start=500
            req = urllib.request.Request(f"https://export.arxiv.org/api/query?search_query={q}&start={start}&max_results={a.per_cat}",
                                         headers={"User-Agent": "ai-detector-bench/1.1 (research)"})
            try:
                xml = urllib.request.urlopen(req, timeout=120).read().decode()
            except Exception as e:  # noqa: BLE001
                print("fail", cat, e, flush=True)
                continue
            for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
                aid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
                title = " ".join(re.search(r"<title>(.*?)</title>", e, re.S).group(1).split())
                abs_ = " ".join(re.search(r"<summary>(.*?)</summary>", e, re.S).group(1).split())
                if aid in have or abs_ in held or len(abs_.split()) < 60:
                    continue
                have.add(aid)
                f.write(json.dumps({"id": f"{aid}#abs", "paper": aid, "cat": cat, "kind": "abstract",
                                    "title": title, "abstract": abs_, "text": abs_}) + "\n")
                n += 1
            print(cat, n, flush=True)
            time.sleep(3)
    print("abstracts added:", n)


if __name__ == "__main__":
    main()
