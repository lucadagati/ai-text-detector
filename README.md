# ai-text-detector

Local service that estimates whether academic text (plain, Markdown or LaTeX) was written by an LLM,
paragraph by paragraph and sentence by sentence, and explains the verdict with the sentence-structure traits
that set the text apart from human papers.

**v4.3 at a glance (numbers from the v4.2 benchmark; not re-run here).** All numbers are at 1% false positives on held-out papers; details in [`bench/RESULTS.md`](bench/RESULTS.md).

| | Claude | AI-polished human text | RAID (11 generators, clean / attacked) | False positives |
|---|---|---|---|---|
| **this detector (OR of three DeBERTa models)** | **93%** (all rewrite modes) | **92%** | **89%** | ≤ 1% (0.2% on 2021 human papers) |
| detector-guided rewrites (adversarial, held-out) | 93-95% | | | ≤ 1% |
| best open detector on RAID (MAGE) | 0.5% | 2% | 69% / 72% | 1% |
| other open detectors (RADAR, academic/ChatGPT/OpenAI RoBERTa) | 0-17% | 0-6% | - | 1% |

Claude models left out of training entirely are still detected at 95%.

## Contents

1. [How it works](#how-it-works)
2. [Install](#install)
3. [Run the server](#run-the-server)
4. [Use the CLI and the API](#use-the-cli-and-the-api)
5. [Revise a manuscript](#revise-a-manuscript)
6. [Humanizer loop](#humanizer-loop)
7. [Tests](#tests)
8. [Benchmark reproduction](#benchmark-and-retraining-bench)
9. [Troubleshooting](#troubleshooting)
10. [Limits](#limits)

**Verification status.** In the environment where this README was last checked (Linux, Python 3.10, one shared GPU) these were run
and worked: `pytest` (10 passed), `ruff check .`, server start and `/api/health`, the web page (`GET /` returns 200), the start of the first-request download of
the model weights from Hugging Face (first model fetched, the rest interrupted), `aidetect` on stdin with `--fail-above` (exit 0 / exit 1, also with `--json`) against a running service, and
`aihumanize --dry-run`. Not verified in this environment: the Docker image, the systemd unit on a fresh host, CPU-only
inference, the `aihumanize` rewrite rounds (they need the `claude` CLI and were not re-run), and every bench script and benchmark number below.

## How it works

1. **Text preparation.** LaTeX is converted to prose: figures, tables, equations, algorithms and listings are
   dropped, and inline math becomes `x`. The markup of the `changes` package (`\added`, `\deleted`, `\replaced`)
   and of `\add`/`\del` is resolved to the accepted text. Paragraphs under 40 words, or dominated by math or code,
   are not assessed.
2. **Classifiers.** Three DeBERTa-v3-base models (mean-pooled, one logit) form an OR ensemble: `deberta-base-v2` is
   trained on our academic data, `deberta-base-v3` adds the RAID training split (base models and attacks), and
   `deberta-base-rev2` adds ~2,900 adversarial paragraphs, AI texts rewritten in loops against the detector itself
   (`bench/generate_adv.py`), and three LLM-written manuscripts before and after a detector-guided revision. A paragraph is flagged if any exceeds its own 0.33%-FPR threshold. The academic data are 4,960 human paragraphs and abstracts
   (pre-2020 arXiv) and 4,718 AI paragraphs from Claude, gpt-oss, Mistral, Qwen, Gemma, Llama and Phi, written from
   scratch or as rewrites of the human paragraphs (polish, "Reviewer 2" revision, multi-pass revision).
   Training also uses random 2-5-sentence windows, so the model scores short passages.
3. **Paragraph labels.** Scores are normalized: `ai` at 1 or above (the 1%-FPR level), `uncertain` at 0 or above
   (5%-FPR), `human` below 0. Text is Unicode-normalized first (NFKC, invisible characters, homoglyphs).
4. **Sentence highlighting.** Each sentence gets the mean score of the 3-sentence windows that contain it.
5. **Document verdict.** At least 3 paragraphs and 30% of them flagged: *likely AI*. At least 3 and 15%:
   *mixed*. This fires on 1 of 90 human papers.
6. **Structural distance.** Independent of the classifiers: the Mahalanobis distance of the document's structure
   profile from human papers (chunks of 12 paragraphs, human p95 5.5, p99 6.75). If the classifiers say human but the
   distance is elevated, the verdict says so, because text revised against a detector can pass the classifiers
   while keeping an unnatural structure. A passing classifier score is not proof of human authorship. On paragraphs mixed from human and AI sentences the share flagged follows the share of AI text (AI paragraph with 75% / 50% / 25% of its sentences kept: 80% / 31% / 8% flagged; human paragraph with 25% AI sentences: 11%), see `bench/RESULTS.md`. Agents revising five unseen manuscripts against this ensemble (5 rewrites per paragraph) left the share of passing paragraphs unchanged for the manuscripts that were mostly flagged (9% before and after).
7. **Explanation.** The document's sentence-structure profile (spaCy parse) is compared with 240 human papers.
   It covers sentence openings, subordination, passive voice, semicolons, ", not X" tails, nominalizations and
   rhythm, and reports the traits farthest from the human norm.

## Install

Needs Python 3.10+, about 3 GB of GPU memory (CPU-only should work but is slower and was not verified) and about 2.2 GB of disk for the
three classifiers (735 MB each). The weights are not in this repository: they are published at
[huggingface.co/Lukkinen/ai-text-detector](https://huggingface.co/Lukkinen/ai-text-detector) and downloaded into `aidet/models/` on the first request
(set `AIDET_HF_REPO` to use another copy).

```bash
git clone https://github.com/lucadagati/ai-text-detector && cd ai-text-detector
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md     # sentence-structure profile
```

The client scripts (`aidetect`, `score_tex.py`, `humanize.py`, `track_changes.py`) use only the standard library; `aidetect` and `aihumanize`
use `.venv/bin/python` if it exists and `python3` otherwise.

## Run the server

```bash
.venv/bin/python server.py                 # http://127.0.0.1:8790
curl -s localhost:8790/api/health          # {"status":"ok","loaded":false,...}
```

The models load on the first `/api/detect` request (about a minute, plus the weight download the very first time) and are released after
15 idle minutes, because the GPU is meant to be shared. The service scores all windows of a document in one length-sorted pass and caches scores
per model. Settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `AIDET_HOST`, `AIDET_PORT` | `127.0.0.1`, `8790` | bind address. The API has no authentication: do not expose it publicly |
| `AIDET_IDLE_MIN` | `15` | idle minutes before the models are unloaded |
| `AIDET_CACHE` | `20000` | cached scores per model |
| `AIDET_MAX_CHARS` | `2000000` | larger requests get HTTP 413 |
| `AIDET_HF_REPO` | `Lukkinen/ai-text-detector` | where the weights are fetched from |
| `AIDET_URL` | `http://127.0.0.1:8790` | used by the clients (CLI, `score_tex.py`, `humanize.py`) |

**As a systemd service:** edit `WorkingDirectory` and `ExecStart` in `ai-detector.service` to your checkout, then
`sudo cp ai-detector.service /etc/systemd/system/ && sudo systemctl enable --now ai-detector` (not verified on a fresh host).

**With Docker (GPU host):** `docker build -t ai-text-detector .` then
`docker run --gpus all -p 127.0.0.1:8790:8790 -v aidet-models:/app/aidet/models ai-text-detector` (not verified).

**Remote use:** forward the port (for example VS Code Remote, or `ssh -L 8790:127.0.0.1:8790 host`) rather than binding to a public address.

## Use the CLI and the API

The CLI talks to a running server and starts nothing itself.

```bash
./aidetect paper/sections/                 # every .tex/.md/.txt in the folder, most AI-like paragraphs first
./aidetect main.tex --top 10 --sentences   # also the most AI-like sentences of flagged paragraphs
./aidetect draft.md --json                 # raw JSON, one object per file
echo "some text" | ./aidetect -            # from stdin
./aidetect paper/sections/ --fail-above 20 # exit 1 if more than 20% of the words are labelled ai (CI / pre-commit gate)

# per paragraph with source line numbers, for revising a manuscript
.venv/bin/python score_tex.py paper/sections/introduction.tex --sentences --only-flagged

# API; the web page at http://127.0.0.1:8790 does the same interactively
curl -s localhost:8790/api/detect -H 'Content-Type: application/json' -d '{"text": "..."}'
```

Paragraphs under 40 words are not assessed (`--min-words` / `min_words` changes that), so a very short text returns `too short`.

### API response

`POST /api/detect` takes `{"text": str, "min_words": 40, "latex": null|bool}` and returns:
- `verdict`, `share_words` (`ai` / `uncertain` / `human`), `n_paragraphs`, and `n_skipped` (math- or code-heavy paragraphs);
- `paragraphs[]`, each with `label`, `score`, `signals` and `sentences[]` (`text`, `score`);
- `structure_distance`, document-level distance of the structure profile from human papers (`distance`, `p95`, `p99`, `level`, `chunks`); when the classifier says human but the level is elevated/high, the verdict says so;
- `structure_traits[]`, the traits farthest from human papers (`trait`, `value`, `human_mean`, `z`);
- `thresholds`, `sentence_threshold` (1% level) and `sentence_threshold_5` (5% level).

## Revise a manuscript


`track_changes.py OLD.tex NEW.tex OUT.tex` writes NEW as a `changes`-package layer (`\added`, `\deleted`,
`\replaced`) over OLD, sentence by sentence. Paragraphs are aligned in order: similar ones are diffed, and
paragraphs that exist only in NEW (or only in OLD) have their prose wrapped in `\added` (or `\deleted`).
Environments and structural commands stay outside the markup; a changed table or figure is written as in NEW
(mark its new cells or caption by hand with `\added`), and `\cite`/`\Cref`/`\ref` in deleted text are protected with `\mbox`.
Citations that survive only in deleted text still need their BibTeX entries until the changes are accepted.
With `\usepackage[final]{changes}` the package expands added text, which fails on `\emph` followed by `\cite`; to print the clean text, redefine `\added`, `\deleted` and `\replaced` instead (accept: `#2`, nothing, `#2`). A typical loop:
1. Score the clean text with `score_tex.py`.
2. Revise the flagged paragraphs and score them again.
3. Generate the tracked file against the original with `track_changes.py`.

## Humanizer loop

`aihumanize FILE -o OUT` (wrapper of `humanize.py`) rewrites the flagged paragraphs with the [humanizer](https://github.com/blader/humanizer) rules
(vendored in `humanizer/`, MIT) through the `claude` CLI and rescores each candidate on the running server, keeping numbers, citations and math fixed.

```bash
./aihumanize paper/sections/intro.tex --dry-run                      # only score, nothing is written
./aihumanize paper/sections/intro.tex -o /tmp/intro.humanized.tex    # rewrite flagged paragraphs, up to 4 rounds each
./aihumanize intro.tex -o out.tex --sample my_text.tex --tracked intro.tracked.tex   # match your voice, add a `changes` layer
```

Options: `--rounds N`, `--model` (default `sonnet`), `--all` (also paragraphs already `human`), `--jobs` (parallel paragraphs, default 3),
`--inplace` (overwrite; incompatible with `--tracked`). Candidates that change numbers, `\cite`/`\ref` keys or inline math, or grow by more than
25%, are discarded. The result is optimized against this detector only: a passing score is not proof of human authorship, so check the structure
distance in `aidetect` output and read the text.

## Tests

```bash
.venv/bin/pip install pytest httpx ruff
.venv/bin/python -m pytest -q      # 10 tests, no GPU or weights needed
.venv/bin/ruff check .
```

CI (`.github/workflows/ci.yml`) runs the same two commands. The tests cover text preparation, paragraph splitting, the humanizer invariants, `score_tex.blocks`,
`track_changes.py`, the server limits (413/422) and the document verdict rules, plus the `--fail-above` gate of the CLI.

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

To reproduce a result, start from `bench/RESULTS.md` (results by version, with the scripts named in the text). Data collection and
training need network access, the `claude` CLI and/or Ollama for generation, and a GPU; none of these scripts was re-run for this README.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `service not reachable at http://127.0.0.1:8790` | the server is not running (`python server.py` or `systemctl start ai-detector`), or `AIDET_URL` is wrong |
| first request takes minutes or fails | the weights are being downloaded (about 2.2 GB) into `aidet/models/`; check network access to huggingface.co or set `AIDET_HF_REPO` |
| CUDA out of memory | other jobs on the GPU; the service needs about 3 GB. Wait for the idle unload, or stop the other job |
| HTTP 413 | text above `AIDET_MAX_CHARS` |
| HTTP 422 | the text could not be parsed as LaTeX (for example unbalanced braces); pass `"latex": false` or fix the source |
| verdict `too short` | no paragraph reached `min_words` (40) after math, figures and tables were removed |
| `No module named spacy` / missing `en_core_web_md` | run the spacy download line of the install step |
| `aihumanize`: `empty output` | the `claude` CLI is missing or not logged in |
| `ModuleNotFoundError: fastapi` for `server.py` | install `requirements.txt` into the same Python that runs it |

## Limits

No detector is proof of authorship. The validation covers English CS/EE academic text. Heavily hand-edited AI
text, or model families unlike the nine used here, can evade it: open-weight models left out of training were
detected at 59%. Isolated flagged paragraphs occur in 8% of human papers and should not be read as evidence.
