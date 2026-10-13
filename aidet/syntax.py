"""Sentence-structure profile of a text (spaCy dependency parse).

Each feature is a rate over sentences (or tokens) of one text; `profile(texts)` aggregates
over a corpus, `paragraph_features(text)` gives the vector used by the structure classifier.
"""
import math
import re
from collections import Counter

import spacy

_nlp = None
SUB = {"advcl", "ccomp", "csubj", "csubjpass", "relcl", "acl"}
NOMINAL = re.compile(r"(tion|sion|ment|ness|ity|ance|ence)s?$")


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_md", disable=["ner"])
    return _nlp


def _depth(tok):
    d = 0
    while tok.head.i != tok.i:  # spaCy builds a new Token object on each access: compare indices
        tok, d = tok.head, d + 1
    return d


def _opening(s):
    toks = [t for t in s if not t.is_punct and not t.is_space]
    if not toks:
        return "other"
    t0 = toks[0]
    if t0.dep_ == "mark" or (t0.pos_ == "SCONJ"):
        return "subordinate clause"
    if t0.lower_ in {"this", "these", "it", "that", "such", "they"}:
        return "anaphoric (This/It/These)"
    if t0.pos_ == "ADP":
        return "prepositional phrase"
    if t0.pos_ == "ADV" or t0.dep_ == "advmod":
        return "adverb / connective"
    if t0.tag_ in {"VBG", "VBN"}:
        return "participle"
    if t0.lower_ in {"we", "our", "i"}:
        return "first person"
    return "subject (noun phrase)"


def sentence_records(text):
    doc = nlp()(text)
    out = []
    for s in doc.sents:
        words = [t for t in s if not t.is_punct and not t.is_space]
        if len(words) < 3:
            continue
        txt = s.text.strip()
        root = s.root
        deps = Counter(t.dep_ for t in s)
        conj_groups = Counter(t.head.i for t in s if t.dep_ == "conj")
        triad = any(n >= 2 for n in conj_groups.values())  # head + >=2 conjuncts = list of 3+
        clause_conj = sum(1 for t in s if t.dep_ == "conj" and t.pos_ in {"VERB", "AUX"})
        n_sub = sum(deps[d] for d in SUB)
        # tail appended after the main clause: ", not X", "; ...", ": ..."
        tail_not = bool(re.search(r",\s+(?:and\s+)?not\s+\w+[^,;:]*[.!?]?$", txt))
        semi = txt.count(";")
        colon = len(re.findall(r"\w: \w", txt))
        out.append({
            "len": len(words),
            "opening": _opening(s),
            "pos3": " ".join(t.pos_ for t in words[:3]),
            "depth": max(_depth(t) for t in s),
            "n_sub": n_sub,
            "simple": n_sub == 0 and clause_conj == 0,
            "clause_conj": clause_conj,
            "passive": deps["nsubjpass"] + deps["auxpass"] > 0,
            "copular": root.lemma_ == "be" or any(c.dep_ in {"attr", "acomp"} and c.head.i == root.i for c in s),
            "appos": deps["appos"] + txt.count("("),
            "triad": triad,
            "tail_not": tail_not,
            "semi": semi > 0,
            "colon": colon > 0,
            "nominal": sum(1 for t in words if t.pos_ == "NOUN" and NOMINAL.search(t.lower_)) / len(words),
            "nv_ratio": sum(t.pos_ == "NOUN" for t in words) / max(1, sum(t.pos_ == "VERB" for t in words)),
            "we": any(t.lower_ in {"we", "our", "us"} for t in words),
        })
    return out


def _agg(recs):
    n = len(recs)
    L = [r["len"] for r in recs]
    mean = sum(L) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in L) / n)
    diffs = [abs(a - b) for a, b in zip(L, L[1:])]
    pos3 = Counter(r["pos3"] for r in recs)
    ent = -sum(c / n * math.log2(c / n) for c in pos3.values())
    f = {
        "sentence length, mean": mean,
        "sentence length, CV": sd / mean,
        "rhythm: |Δlength| between consecutive sentences / mean": (sum(diffs) / len(diffs) / mean) if diffs else 0.0,
        "short sentences (<10 words) %": 100 * sum(x < 10 for x in L) / n,
        "long sentences (>35 words) %": 100 * sum(x > 35 for x in L) / n,
        "simple sentences (no subordinate/coordinate clause) %": 100 * sum(r["simple"] for r in recs) / n,
        "subordinate clauses per sentence": sum(r["n_sub"] for r in recs) / n,
        "coordinated clauses per sentence": sum(r["clause_conj"] for r in recs) / n,
        "parse-tree depth, mean": sum(r["depth"] for r in recs) / n,
        "passive %": 100 * sum(r["passive"] for r in recs) / n,
        "copular 'X is Y' %": 100 * sum(r["copular"] for r in recs) / n,
        "appositives/parentheses per sentence": sum(r["appos"] for r in recs) / n,
        "lists of 3+ items %": 100 * sum(r["triad"] for r in recs) / n,
        "ends with ', not X' %": 100 * sum(r["tail_not"] for r in recs) / n,
        "contains ';' %": 100 * sum(r["semi"] for r in recs) / n,
        "contains ': ' mid-sentence %": 100 * sum(r["colon"] for r in recs) / n,
        "first person in sentence %": 100 * sum(r["we"] for r in recs) / n,
        "nominalizations per word %": 100 * sum(r["nominal"] for r in recs) / n,
        "noun/verb ratio": sum(r["nv_ratio"] for r in recs) / n,
        "opening-pattern entropy (bits, first 3 POS)": ent,
    }
    op = Counter(r["opening"] for r in recs)
    for k in ["subject (noun phrase)", "anaphoric (This/It/These)", "prepositional phrase",
              "subordinate clause", "adverb / connective", "participle", "first person"]:
        f[f"opens with {k} %"] = 100 * op[k] / n
    return f


def profile(texts):
    recs = [r for t in texts for r in sentence_records(t)]
    return _agg(recs), len(recs)


def paragraph_features(text):
    recs = sentence_records(text)
    return _agg(recs) if len(recs) >= 2 else None
