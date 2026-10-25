"""Neutral benchmark: the 'abstracts' domain of RAID (Dugan et al., ACL 2024, liamdugan/raid on Hugging Face).

Downloads the train parquet shards one at a time, keeps domain == abstracts, and samples
human abstracts, clean AI abstracts (no attack) and attacked AI abstracts, balanced over models and attacks.
Rows are split by title (the paper an abstract belongs to): 70% train, 10% calibration, 20% test.
Output: bench/data/raid_abstracts.jsonl  {id, y, text, src="raid|<model>|<attack>", paper, part}
"""
import json, os, random, tempfile, urllib.request, zlib

import pyarrow.parquet as pq

D = os.path.join(os.path.dirname(__file__), "data")
BASE = "https://huggingface.co/api/datasets/liamdugan/raid/parquet/raid/train"
N_HUMAN, N_CLEAN, N_ATTACK = 6000, 6000, 6000


def main():
    shards = json.loads(urllib.request.urlopen(BASE, timeout=60).read())
    human, clean, attacked = [], [], []
    for url in shards:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            t = pq.read_table(tmp.name, columns=["id", "model", "attack", "domain", "title", "generation"])
            t = t.filter(t["domain"].to_numpy(zero_copy_only=False) == "abstracts").to_pylist()
        for r in t:
            if not r["generation"] or len(r["generation"].split()) < 40:
                continue
            (human if r["model"] == "human" else clean if r["attack"] == "none" else attacked).append(r)
        print(url.rsplit("/", 1)[1], len(human), len(clean), len(attacked), flush=True)
    rnd = random.Random(0)

    def balanced(rows, n, key):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        per = n // len(groups)
        return [r for g in groups.values() for r in rnd.sample(g, min(per, len(g)))]

    def part(title):
        u = zlib.crc32((title or "").encode()) % 100
        return "train" if u < 70 else ("calib" if u < 80 else "test")

    out = []
    for rows, y in ((rnd.sample(human, min(N_HUMAN, len(human))), 0),
                    (balanced(clean, N_CLEAN, lambda r: r["model"]), 1),
                    (balanced(attacked, N_ATTACK, lambda r: r["attack"]), 1)):
        for r in rows:
            out.append({"id": r["id"], "y": y, "text": " ".join(r["generation"].split()),
                        "src": f"raid|{r['model']}|{r['attack']}", "paper": "raid:" + (r["title"] or r["id"])[:200],
                        "part": part(r["title"])})
    with open(os.path.join(D, "raid_abstracts.jsonl"), "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("written", len(out))


if __name__ == "__main__":
    main()
