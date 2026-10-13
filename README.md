# ai-text-detector

Local service that estimates whether academic text (plain, Markdown or LaTeX) was written by an LLM,
paragraph by paragraph and sentence by sentence, and explains the verdict with the sentence-structure traits
that set the text apart from human papers.

**v3 at a glance.** All numbers are at 1% false positives on held-out papers; details in [`bench/RESULTS.md`](bench/RESULTS.md).

| | Claude | AI-polished human text | RAID (11 generators, clean / attacked) | False positives |
|---|---|---|---|---|
| **this detector (OR of two DeBERTa models)** | **97%** | **93%** | **80% / 83%** | ≤ 1% |
| best open detector on RAID (MAGE) | 0.5% | 2% | 69% / 72% | 1% |
| other open detectors (RADAR, academic/ChatGPT/OpenAI RoBERTa) | 0-17% | 0-6% | - | 1% |

Claude models left out of training entirely are still detected at 95%.

## How it works

1. **Text preparation.** LaTeX is converted to prose: figures, tables, equations, algorithms and listings are
   dropped, and inline math becomes `x`. The markup of the `changes` package (`\added`, `\deleted`, `\replaced`)
   and of `\add`/`\del` is resolved to the accepted text. Paragraphs under 40 words, or dominated by math or code,
   are not assessed.
2. **Classifiers.** Two DeBERTa-v3-base models (mean-pooled, one logit) form an OR ensemble: `deberta-base-v2` is
   trained on our academic data, and `deberta-base-v3` adds the RAID training split (base models and attacks). A
   paragraph is flagged if either exceeds its own 0.5%-FPR threshold. The academic data are 4,960 human paragraphs and abstracts
   (pre-2020 arXiv) and 4,718 AI paragraphs from Claude, gpt-oss, Mistral, Qwen, Gemma, Llama and Phi, written from
   scratch or as rewrites of the human paragraphs (polish, "Reviewer 2" revision, multi-pass revision).
   Training also uses random 2-5-sentence windows, so the model scores short passages.
3. **Paragraph labels.** Scores are normalized: `ai` at 1 or above (the 1%-FPR level), `uncertain` at 0 or above
   (5%-FPR), `human` below 0. Text is Unicode-normalized first (NFKC, invisible characters, homoglyphs).
4. **Sentence highlighting.** Each sentence gets the mean score of the 3-sentence windows that contain it.
5. **Document verdict.** At least 3 paragraphs and 30% of them flagged: *likely AI*. At least 3 and 15%:
   *mixed*. This fires on 1 of 90 human papers.
6. **Explanation.** The document's sentence-structure profile (spaCy parse) is compared with 240 human papers.
   It covers sentence openings, subordination, passive voice, semicolons, ", not X" tails, nominalizations and
   rhythm, and reports the traits farthest from the human norm.

## Install

Needs Python 3.10+ and about 2 GB of GPU memory (it also runs on CPU, more slowly). The two trained classifiers
(735 MB each) are published at [huggingface.co/Lukkinen/ai-text-detector](https://huggingface.co/Lukkinen/ai-text-detector)
and downloaded into `aidet/models/` on the first request (set `AIDET_HF_REPO` to use another copy).

```bash
git clone https://github.com/lucadagati/ai-text-detector && cd ai-text-detector
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md
.venv/bin/python server.py                       # http://127.0.0.1:8790
# as a service: edit the paths in ai-detector.service, then
sudo cp ai-detector.service /etc/systemd/system/ && sudo systemctl enable --now ai-detector
```

## Use

```bash
# web page: open http://127.0.0.1:8790 (from VS Code Remote, forward port 8790)

# command line
./aidetect paper/sections/           # every .tex/.md/.txt in the folder
./aidetect main.tex --top 10
./aidetect draft.md --json

# per paragraph, with source line numbers and the most AI-like sentences (for revising a manuscript)
.venv/bin/python score_tex.py paper/sections/introduction.tex --sentences --only-flagged

# API
curl -s localhost:8790/api/detect -H 'Content-Type: application/json' -d '{"text": "..."}'
```

The model loads on the first request and is released after 15 idle minutes (`AIDET_IDLE_MIN`).
The CLI and `score_tex.py` talk to the service at `AIDET_URL` (default `http://127.0.0.1:8790`).

### API response

`POST /api/detect` takes `{"text": str, "min_words": 40, "latex": null|bool}` and returns:
- `verdict`, `share_words` (`ai` / `uncertain` / `human`), `n_paragraphs`, and `n_skipped` (math- or code-heavy paragraphs);
- `paragraphs[]`, each with `label`, `score`, `signals` and `sentences[]` (`text`, `score`);
- `structure_traits[]`, the traits farthest from human papers (`trait`, `value`, `human_mean`, `z`);
- `thresholds`, `sentence_threshold` (1% level) and `sentence_threshold_5` (5% level).

### Remote use

The service listens on `127.0.0.1` only. From another machine, forward the port with VS Code Remote
or `ssh -L 8790:localhost:8790 <host>`, then use `http://localhost:8790` and set `AIDET_URL` if needed.
To serve a private network instead, set `AIDET_HOST` (for example, a Tailscale address). The API has no
authentication, so do not expose it publicly.

### Revising a manuscript with tracked changes

`track_changes.py OLD.tex NEW.tex OUT.tex` writes NEW as a `changes`-package layer (`\added`, `\deleted`,
`\replaced`) over OLD, sentence by sentence. Paragraphs are aligned in order: similar ones are diffed, and
paragraphs that exist only in NEW (or only in OLD) have their prose wrapped in `\added` (or `\deleted`).
Environments and structural commands stay outside the markup; a changed table or figure is written as in NEW
(mark its new cells or caption by hand with `\added`), and `\cite`/`\Cref`/`\ref` in deleted text are protected with `\mbox`.
Citations that survive only in deleted text still need their BibTeX entries until the changes are accepted.
With `\\usepackage[final]{changes}` the package expands added text, which fails on `\\emph` followed by `\\cite`; to print the clean text, redefine `\\added`, `\\deleted` and `\\replaced` instead (accept: `#2`, nothing, `#2`). A typical loop:
1. Score the clean text with `score_tex.py`.
2. Revise the flagged paragraphs and score them again.
3. Generate the tracked file against the original with `track_changes.py`.

## Benchmark and retraining (`bench/`)

| Script | Role |
|---|---|
| `collect_human.py`, `collect_abstracts.py` | human paragraphs and abstracts from pre-2020 arXiv |
| `generate_ai.py` | AI counterparts via `claude -p` (batched) and Ollama, six modes |
| `finetune.py` | trains the classifier, split by paper; `--exclude` tests unseen generators |
| `features.py`, `structure.py`, `stack.py` | off-the-shelf detectors, structure profile, signal selection, thresholds, `aidet/model.json` |
| `sanity.py` | shortcut checks: held-out abstracts, LaTeX artifacts, given manuscripts |
| `fetch_raid.py` | RAID abstracts (neutral public benchmark), split by paper |
| `external.py` | comparison with open detectors (RADAR, MAGE, RoBERTa detectors) on the same texts |
| `combine.py`, `make_or_model.py` | OR ensembles and stacking at a fixed FPR; writes the OR `aidet/model.json` |
| `eval_lynote.py` | comparison with the lynote-ai lightweight analyzer |

The benchmark texts are not redistributed, because they contain arXiv content. The scripts rebuild them.

## Limits

No detector is proof of authorship. The validation covers English CS/EE academic text. Heavily hand-edited AI
text, or model families unlike the nine used here, can evade it: open-weight models left out of training were
detected at 59%. Isolated flagged paragraphs occur in 8% of human papers and should not be read as evidence.
