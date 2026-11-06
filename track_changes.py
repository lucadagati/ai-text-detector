"""Write a `changes`-package tracked version of NEW against OLD, sentence by sentence.

  track_changes.py OLD.tex NEW.tex OUT.tex
Paragraphs are aligned in order: identical ones are copied, similar ones are diffed, and paragraphs present
only in NEW (or only in OLD) have their prose wrapped in \\added (or \\deleted). Environments (equation,
figure, table, ...) and structural commands (\\section, \\paragraph, \\label, \\item, enumerate, abstract)
stay outside the markup; an environment that changed is written as in NEW, unmarked (mark its new cells by
hand with \\added in NEW if needed), and a deleted one is dropped.
Inside prose, unchanged sentences are copied, changed runs become \\replaced{new}{old} trimmed to the differing
words, and pure insertions/deletions become \\added / \\deleted. In deleted text, \\cite, \\Cref, \\cref and \\ref
are wrapped in \\mbox{} (ulem cannot strike them otherwise).
"""
import difflib, re, sys

ENV = r"\\begin\{(?P<env>equation|align|figure|table|tikzpicture|tabularx|algorithm)\*?\}.*?\\end\{(?P=env)\*?\}"
STRUCT = (r"\\section\*?\{[^}]*\}|\\subsection\*?\{[^}]*\}|\\paragraph\{[^}]*\}|\\label\{[^}]*\}|\\item\b|"
          r"\\begin\{enumerate\}(?:\[[^\]]*\])?|\\end\{enumerate\}|\\begin\{abstract\}|\\end\{abstract\}")
UNIT = re.compile(f"({ENV}|{STRUCT}|\\n)", re.S)


def paragraphs(t):
    # split on blank lines that are not inside an environment
    out, cur, depth = [], [], 0
    for line in t.split("\n"):
        depth += len(re.findall(r"\\begin\{(?:equation|align|figure|table|tikzpicture|tabularx)\*?\}", line))
        depth -= len(re.findall(r"\\end\{(?:equation|align|figure|table|tikzpicture|tabularx)\*?\}", line))
        if not line.strip() and depth <= 0:
            out.append("\n".join(cur)); cur = []
        else:
            cur.append(line)
    out.append("\n".join(cur))
    return out


def pieces(p):
    """alternating prose / unit list; units are environments, structural commands and newlines"""
    parts, last = [], 0
    for m in UNIT.finditer(p):
        parts.append(("prose", p[last:m.start()]))
        parts.append(("unit", m.group(0)))
        last = m.end()
    parts.append(("prose", p[last:]))
    return parts


def balanced(s):
    d = 0
    for c in s:
        d += c == "{"
        d -= c == "}"
        if d < 0:
            return False
    return d == 0 and s.count("$") % 2 == 0


def sentences(s):
    """sentences and the whitespace between them, as alternating list items (joins back to s)"""
    out, cur, depth, math, i = [], "", 0, False, 0
    while i < len(s):
        c = s[i]
        cur += c
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "$":
            math = not math
        elif c in ".!?" and depth == 0 and not math:
            ws = re.match(r"\s+(?=[A-Z\\(])", s[i + 1:])
            if ws:
                out += [cur, ws.group(0)]
                cur = ""
                i += 1 + len(ws.group(0))
                continue
        i += 1
    if cur:
        out.append(cur)
    return out


def protect(old):
    return re.sub(r"(\\(?:cite|Cref|cref|ref|eqref)\{[^}]*\})", r"\\mbox{\1}", old)


def mark(old, new):
    if old == new:
        return old
    if not old.strip():
        return old + r"\added{" + new.strip() + "}" + (" " if new.endswith(" ") else "")
    if not new.strip():
        return r"\deleted{" + protect(old.strip()) + "}"
    ow, nw = old.split(" "), new.split(" ")
    # longest common prefix/suffix, then back off to the longest one that is balanced ({} and $)
    lp = 0
    while lp < min(len(ow), len(nw)) and ow[lp] == nw[lp]:
        lp += 1
    k = max([i for i in range(lp + 1) if balanced(" ".join(ow[:i]))])
    ls = 0
    while ls < min(len(ow), len(nw)) - k and ow[-1 - ls] == nw[-1 - ls]:
        ls += 1
    j = max([i for i in range(ls + 1) if balanced(" ".join(ow[len(ow) - i:]) if i else "")])
    pre = " ".join(ow[:k]); suf = " ".join(ow[len(ow) - j:]) if j else ""
    om = " ".join(ow[k:len(ow) - j]); nm = " ".join(nw[k:len(nw) - j])
    if not balanced(om) or not balanced(nm):  # fall back to the whole sentence run
        pre, suf, om, nm = "", "", old, new
    if not om.strip():
        body = r"\added{" + nm + "}"
    elif not nm.strip():
        body = r"\deleted{" + protect(om) + "}"
    else:
        body = r"\replaced{" + nm + "}{" + protect(om) + "}"
    return " ".join(x for x in (pre, body, suf) if x)


def diff_prose(old, new):
    if old == new:
        return old
    so, sn = sentences(old), sentences(new)
    out = []
    sm = difflib.SequenceMatcher(a=so, b=sn, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        if op == "equal":
            out.append("".join(so[a0:a1]))
        else:
            o, n = "".join(so[a0:a1]), "".join(sn[b0:b1])
            lead = re.match(r"\s*", o if o.strip() else n).group(0)
            trail = o[len(o.rstrip()):] if o.strip() else (n[len(n.rstrip()):] if n.strip() else "")
            out.append(lead + mark(o.strip(), n.strip()) + trail)
    return "".join(out)


def similarity(a, b):
    return difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False).ratio()


def align(po, pn, min_ratio=0.3):
    """(old, new) pairs in order; None marks a paragraph present on one side only"""
    out = []
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=po, b=pn, autojunk=False).get_opcodes():
        if op == "equal":
            out += list(zip(po[a0:a1], pn[b0:b1]))
            continue
        i = a0
        for j in range(b0, b1):
            best = max(range(i, a1), key=lambda k: similarity(po[k], pn[j]), default=None)
            if best is not None and similarity(po[best], pn[j]) >= min_ratio:
                out += [(po[k], None) for k in range(i, best)] + [(po[best], pn[j])]
                i = best + 1
            else:
                out.append((None, pn[j]))
        out += [(po[k], None) for k in range(i, a1)]
    return out


def whole(p, side):
    """a paragraph present on one side only: its prose wrapped, its units kept (new) or dropped (old)"""
    res = []
    for k, t in pieces(p):
        if k == "unit":
            res.append(t if side == "new" else "")
        elif t.strip():
            lead, trail = t[:len(t) - len(t.lstrip())], t[len(t.rstrip()):]
            body = r"\added{" + t.strip() + "}" if side == "new" else r"\deleted{" + protect(t.strip()) + "}"
            res.append(lead + body + trail)
        else:
            res.append(t)
    return "".join(res)


HEAD = re.compile(r"\\paragraph\{([^}]*)\}[ \n]?")


def diff_paragraph(a, b):
    # a run-in heading removed (or added) in front of an otherwise similar paragraph
    ha, hb = HEAD.match(a), HEAD.match(b)
    if ha and not hb:
        return r"\deleted{" + ha.group(1) + ".} " + diff_paragraph(a[ha.end():], b)
    if hb and not ha:
        return b[:hb.end()] + diff_paragraph(a, b[hb.end():])
    xa, xb = pieces(a), pieces(b)
    ua = [t for k, t in xa if k == "unit"]; ub = [t for k, t in xb if k == "unit"]
    if ua == ub:
        return "".join(t if k == "unit" else diff_prose(t, tb) for (k, t), (_, tb) in zip(xa, xb))
    # units differ: align them; prose between aligned units is diffed, changed units are taken from NEW
    pa = [t for k, t in xa if k == "prose"]; pb = [t for k, t in xb if k == "prose"]
    res = [diff_prose(pa[0], pb[0])]
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=ua, b=ub, autojunk=False).get_opcodes():
        if op == "equal":
            for i, j in zip(range(a0, a1), range(b0, b1)):
                res += [ub[j], diff_prose(pa[i + 1], pb[j + 1])]
            continue
        old_prose = "".join(pa[i + 1] for i in range(a0, a1))
        for j in range(b0, b1):
            res.append(ub[j])
            res.append(diff_prose(old_prose, pb[j + 1]) if j == b1 - 1 else mark("", pb[j + 1]) if pb[j + 1].strip() else pb[j + 1])
        if b0 == b1 and old_prose.strip():
            res.append(mark(old_prose, ""))
    return "".join(res)


def main():
    old, new, out = sys.argv[1:4]
    po, pn = paragraphs(open(old).read().rstrip()), paragraphs(open(new).read().rstrip())
    res = []
    for a, b in align(po, pn):
        if a == b:
            res.append(a)
        elif a is None:
            res.append(whole(b, "new"))
        elif b is None:
            d = whole(a, "old")
            if d.strip():
                res.append(d)
        else:
            res.append(diff_paragraph(a, b))
    open(out, "w").write("\n\n".join(res) + "\n")


if __name__ == "__main__":
    main()
