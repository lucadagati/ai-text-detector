"""Deterministic checks for the academic-writing guidelines (humanizer/ACADEMIC.md).

lint(text) -> dict of counts; violations(old, new) -> list of reasons a rewrite must be rejected.
Rules: do not add stock formulas, hype words or contractions; do not lose technical terms, acronyms or \\texttt/\\emph terms
(no synonym swapping); do not lengthen the mean sentence by >35% (simplify, never pad).
"""
import re, statistics, collections

FORMULAS = [
    r"it is (?:important|worth|useful|interesting) to (?:note|emphasi[sz]e|highlight|mention|stress|underline)",
    r"(?:it is )?worth (?:noting|mentioning|remembering)", r"it should be (?:noted|emphasi[sz]ed)",
    r"plays? an? (?:crucial|key|pivotal|vital|fundamental|important|significant|essential) role",
    r"in today'?s (?:\w+ )?(?:landscape|world|era)", r"in the (?:current|modern|present) (?:\w+ )?landscape",
    r"(?:numerous|countless|myriad) (?:benefits|advantages|applications)", r"a (?:wide|broad) (?:range|array|variety) of",
    r"not only\b.{1,120}?\bbut also\b", r"in this context", r"in the realm of", r"at the heart of",
    r"paves? the way", r"sheds? (?:new )?light on", r"a testament to", r"game[- ]chang", r"shap(?:es|ing) the future",
    r"\b(?:moreover|furthermore|additionally),", r"\bdelve[sd]?\b", r"\btapestry\b", r"\bunderscores?\b",
    r"\bleverag(?:e|es|ed|ing)\b", r"\bseamless(?:ly)?\b", r"\bcutting-edge\b", r"\btransformative\b", r"\bgroundbreaking\b",
    r"\bholistic\b", r"\bparadigm shift\b", r"\bpivotal\b", r"\bnotably\b", r"\bcrucial(?:ly)?\b",
]
HYPE = [r"\bremarkabl[ey]\b", r"\bdramatic(?:ally)?\b", r"\bsignificantly\b", r"\bstate-of-the-art\b", r"\bnovel\b",
        r"\binnovative\b", r"\brobust(?:ness)?\b", r"\bpowerful\b", r"\bsuperior\b", r"\bunprecedented\b", r"\bcompelling\b"]
CONTRACTION = r"\b\w+(?:n't|'re|'ve|'ll|'d)\b"
FORMULAS_RE = [re.compile(p, re.I | re.S) for p in FORMULAS]
HYPE_RE = [re.compile(p, re.I) for p in HYPE]


def _prose(t):
    t = re.sub(r"\\(?:cite|citep|citet|ref|Cref|cref|eqref|label)\w*\{[^}]*\}", " ", t)
    t = re.sub(r"\$[^$]+\$", " ", t)
    return t


def _sentences(t):
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", _prose(t))
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", t.strip()) if len(s.split()) > 2]


def _terms(t):
    """Terms that must survive a rewrite: acronyms, identifiers with _ or digits, \\texttt/\\emph/\\textit contents."""
    p = _prose(t)
    acr = re.findall(r"\b[A-Z][A-Za-z]*[A-Z][A-Za-z0-9]*\b", p)
    ident = re.findall(r"\b\w+_\w+\b", p)
    fmt = re.findall(r"\\(?:texttt|emph|textit|textsc)\{([^}]*)\}", t)
    return collections.Counter(a for a in acr + ident + [f.strip() for f in fmt] if a)


def lint(text):
    p = _prose(text)
    sents = _sentences(text)
    lens = [len(s.split()) for s in sents]
    openings = collections.Counter(" ".join(s.split()[:2]).lower() for s in sents)
    return {
        "formulas": sum(len(r.findall(p)) for r in FORMULAS_RE),
        "hype": sum(len(r.findall(p)) for r in HYPE_RE),
        "contractions": len(re.findall(CONTRACTION, p)),
        "mean_len": statistics.mean(lens) if lens else 0.0,
        "len_cv": (statistics.pstdev(lens) / statistics.mean(lens)) if len(lens) > 2 else None,
        "repeat_openings": sum(v - 1 for v in openings.values() if v > 1),
    }


def violations(old, new):
    a, b, out = lint(old), lint(new), []
    if b["formulas"] > a["formulas"]:
        out.append("added stock formulas")
    if b["hype"] > a["hype"]:
        out.append("added hype words")
    if b["contractions"] > a["contractions"]:
        out.append("added contractions")
    lost = _terms(old) - _terms(new)
    if lost:
        out.append("terminology changed: " + ", ".join(sorted(lost)[:4]))
    if a["mean_len"] and b["mean_len"] > 1.35 * a["mean_len"]:
        out.append("sentences got longer")
    return out


if __name__ == "__main__":
    import sys, json
    print(json.dumps(lint(open(sys.argv[1], encoding="utf-8").read()), indent=1))
