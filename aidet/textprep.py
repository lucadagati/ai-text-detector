"""LaTeX/Markdown to plain prose, resolving `changes` markup to the accepted text."""
import re
import unicodedata

# invisible characters and Cyrillic/Greek look-alikes used to evade detectors (homoglyph attacks)
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_HOMOGLYPH = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ѕ": "s",
    "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ν": "v", "ο": "o", "α": "a",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I",
    "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
})


def normalize_unicode(t):
    """NFKC, drop invisible characters, map look-alike letters to Latin ones."""
    return unicodedata.normalize("NFKC", t).translate(_INVISIBLE).translate(_HOMOGLYPH)

def _braced(s, i):
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def accept_changes(s):
    """Keep added/new text, drop deleted/old text (also for \\add / \\del)."""
    out, i = [], 0
    cmds = (("\\replaced", 2), ("\\added", 1), ("\\deleted", 1), ("\\add", 1), ("\\del", 1))
    while i < len(s):
        for cmd, nargs in cmds:
            if s.startswith(cmd + "{", i) or s.startswith(cmd + "[", i):
                k = i + len(cmd)
                if s[k] == "[":  # optional [id=..] argument
                    k = s.index("]", k) + 1
                a, k = _braced(s, k)
                if cmd == "\\replaced":
                    _, k = _braced(s, k)
                if cmd in ("\\replaced", "\\added", "\\add"):
                    out.append(accept_changes(a))
                i = k
                break
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def latex_to_text(t):
    t = accept_changes(normalize_unicode(t))
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    t = re.sub(r"\\begin\{(figure|table|tikzpicture|algorithm|algorithmic|lstlisting|verbatim|minted)\*?\}.*?\\end\{\1\*?\}",
               "\n\n", t, flags=re.S)
    t = re.sub(r"\\begin\{(equation|align|gather)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"~?\\[cC]ite[pt]?\{[^}]*\}", "", t)
    t = re.sub(r"\\[cC]ref\{[^}]*\}", "Section 3", t)
    t = re.sub(r"\\ref\{[^}]*\}", "3", t)
    t = re.sub(r"\\(section|subsection|paragraph)\*?\{([^}]*)\}", r"\n\n\2.\n\n", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    for _ in range(3):
        t = re.sub(r"\\(emph|texttt|textbf|textit|mbox|url)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\$[^$]*\$", "x", t)                       # inline math -> placeholder
    t = t.replace("\\,", " ").replace("~", " ").replace("--", "-").replace("``", '"').replace("''", '"')
    t = re.sub(r"\\item\b", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t)
    t = t.replace("{", "").replace("}", "")
    return t


def looks_like_latex(t):
    return bool(re.search(r"\\(section|begin|cite|emph|textbf|paragraph|label|ref|Cref)\b", t))


def assessable(p):
    """Same filter as the training data: mostly prose, not dominated by math placeholders."""
    w = p.split()
    return sum(c.isalpha() for c in p) / max(1, len(p)) >= 0.75 and p.count("x") / max(1, len(w)) <= 0.15


def split_paragraphs(text, min_words=40, latex=None, with_skipped=False):
    if latex is None:
        latex = looks_like_latex(text)
    text = latex_to_text(text) if latex else normalize_unicode(text)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if len(p.split()) >= min_words]
    keep = [p for p in paras if assessable(p)]
    return (keep, len(paras) - len(keep)) if with_skipped else keep


