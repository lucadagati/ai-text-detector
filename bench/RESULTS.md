# Benchmark results (v3, 2026-10-12)

All numbers are on **held-out papers**: the split is by paper (60% train, 15% calibration, 25% test), so no
paragraph of a test paper, nor any AI rewrite of it, is seen in training. Thresholds are fixed on the
calibration papers and then applied to the test papers.

## Data

| Class | Source | Size |
|---|---|---|
| Human | body paragraphs of 242 arXiv papers from 2015-2019 (LaTeX sources, 20 CS/EE categories) | 2,963 |
| Human | arXiv abstracts from 2015-2019 | 1,997 |
| AI | Claude Sonnet 957, Opus 560, Haiku 446; gpt-oss-20b 521, mistral-small-24b 497, gemma3-12b 462, qwen2.5-14b 446, llama3.1-8b 437, phi4-14b 392 | 4,718 |

AI modes: `free` (new paragraph or abstract for the same paper) 1,101; `human` (instructed to sound human) 1,102;
`polish` (Q1-editor rewrite of the human paragraph) 1,151; `revise` ("Reviewer 2" revision without LLM clichés) 965;
`results` (quantitative results paragraph) 199; `multipass` (three successive editorial revisions) 200.
Every text goes through the same LaTeX-to-prose pipeline, and math- or code-heavy paragraphs are excluded on both sides.

## Paragraph level (test papers: 1,126 human, 1,065 AI)

| Detector | AUROC | AI detected at 1% FPR |
|---|---|---|
| **v2: fine-tuned DeBERTa-v3-base** | **0.998** | **97.3%** |
| v1 ensemble (Binoculars + desklib + fakespot) | 0.878 | 44% |
| desklib (off the shelf) | 0.800 | 33% |
| fakespot (off the shelf) | 0.816 | 38% |
| sentence-structure profile alone | 0.839 | 14% |
| Binoculars, Falcon-7B pair (v1 benchmark) | 0.563 | 13% |
| lynote-ai lightweight analyzer (v1 benchmark) | 0.563 | 4% at 5% FPR |

Combining the fine-tuned model with desklib, fakespot or the structure profile changes detection by at most
0.5 points in distribution, and lowers it on out-of-distribution text (a fully AI-written journal manuscript:
46% of paragraphs flagged by the classifier alone, 24% by the ensemble). v2 therefore uses the classifier alone;
the structure profile is kept as an explanation.

At the calibrated 1%-FPR threshold:

| Source | Detected | Source | Detected |
|---|---|---|---|
| human body paragraphs (false positives) | 0.7% | human abstracts (false positives) | 0.4% |
| Claude Haiku | 100% | gpt-oss-20b | 100% |
| Claude Opus | 96.9% | gemma3-12b | 100% |
| Claude Sonnet | 96.4% | qwen2.5-14b | 96.1% |
| phi4-14b | 95.0% | mistral-small-24b | 93.3% |
| llama3.1-8b | 82.2% | | |
| mode `free` | 97.9% | mode `human` (adversarial) | 98.5% |
| mode `polish` | 91.8% | mode `revise` | 95.2% |
| mode `results` | 95.8% | mode `multipass` | 96.6% |

## Unseen generators

Retrained without one family, tested on that family's test paragraphs (1% FPR):

| Left out of training | Detected |
|---|---|
| all Claude models | 95.2% |
| gpt-oss-20b | 94.9% |
| all open-weight models (llama, mistral, qwen, gemma, phi4) | 58.9% |

Detection carries over to unseen frontier models. Small open-weight models are a different style: a new
family of that kind should be added to the training data.

## Shortcut checks (bench/sanity.py)

- 400 held-out pre-2020 arXiv abstracts, never seen in training: 0.5% flagged.
- LaTeX-conversion artifacts (formula placeholders) do not drive the score: injecting them into AI text
  moves detection from 96.3% to 93.8%; removing them from human text leaves false positives at 0.3%.

## Document level

Rule "at least 3 paragraphs and at least 15% of them flagged" → *mixed*. "At least 30%" → *likely AI*.
It fires on 1 of 90 human papers (≥ 5 paragraphs). A single flagged paragraph occurs in 8% of human papers,
and is reported as inconclusive.

## v3: comparison with open detectors and the OR ensemble

**Detectors compared.** We ran five open detectors from other groups on the same texts:
- RADAR (TrustSafeAI/RADAR-Vicuna-7B, trained against paraphrasing);
- MAGE (yaful/MAGE);
- an academic RoBERTa (andreas122001/roberta-academic-detector);
- the HC3 ChatGPT detector;
- OpenAI's RoBERTa-large detector.

**Neutral benchmark.** The abstracts domain of RAID (Dugan et al., ACL 2024): 11 generators from GPT-2 to GPT-4, base
and chat models, and 11 attacks (paraphrase, synonym, homoglyph, zero-width space, misspellings, ...). It is split by
paper: train 70%, calibration 10%, test 20%.

**Thresholds.** Every detector is held at 1% false positives on the same calibration humans (our papers plus RAID).

| Detector | Our test AUROC | Claude | AI rewrites | RAID clean | RAID attacked | FP held-out abstracts | LD2 flagged |
|---|---|---|---|---|---|---|---|
| RADAR | 0.54 | 17% | 4% | - | - | 2.0% | 0% |
| MAGE | 0.51 | 0.5% | 2% | 69% | 72% | 1.0% | 0% |
| academic RoBERTa | 0.69 | 3% | 6% | - | - | 3.0% | 0% |
| HC3 ChatGPT RoBERTa | 0.68 | 0% | 0% | - | - | 0% | 0% |
| OpenAI RoBERTa | 0.69 | 7% | 3% | - | - | 2.8% | 0% |
| ours v2 (our data only) | 0.998 | 98% | 95% | 28% | 19% | 0.7% | 52% |
| ours v3 (+ RAID train) | 0.99 | 90% | 80% | 81% | 84% | 0.7% | 36% |
| **ours, OR(v2, v3)** | - | **97%** | **93%** | **80%** | **83%** | 1.0% | **52%** |

On RAID, MAGE reaches AUROC 0.95 and ours v2 0.79 when each is measured alone. The OR ensemble flags a paragraph
if either model exceeds its own 0.5%-FPR threshold, so the overall false-positive rate stays at 1% or below. In
calibration it gives 0.4% on our papers, 1.0% on RAID humans and 1.0% on held-out abstracts. It keeps v2's
detection on modern models and academic text, and beats MAGE on RAID by about 11 points (clean and attacked).

**Fixes in v3.**
- Unicode normalization (NFKC, invisible characters removed, Cyrillic and Greek look-alikes mapped to Latin) removes the
  homoglyph and zero-width-space attacks, which every detector missed entirely (0%).
- Before RAID training, v2 missed the RAID base (non-chat) models (GPT-2, GPT-3, Mistral, MPT, Cohere: 2-16%) and
  paraphrase attacks (2%). That is why v3 adds RAID's training split.

**Still open.** Detector-guided rewriting: our LD1 revision, rewritten until v2 scored it human, is flagged 0% by every
detector except RADAR (9%). Training against such rewrites is the next step.

## Limits

- English academic CS/EE text only. Human text is pre-2020 arXiv; journal-edited human prose may differ.
- AI text from 9 models (2024-2026). Heavily hand-edited AI text, or new model families unlike these, can evade it.
- 46-49% of the paragraphs of a manuscript written entirely with Claude over many revision rounds are flagged:
  enough for a clear document verdict, but individual paragraphs of such text often pass.
- No detector is proof of authorship.
