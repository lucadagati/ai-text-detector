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

## Document-level structural distance (2026-10-23)

`bench/doc_structure.py` -> `aidet/docnorm.json`, used by `structure_distance()` in `aidet/ensemble.py`.
Independent of the classifier, so a text revised until the classifier passes it is still caught.

- Profile of 21 structure features (sentence length/rhythm, long and simple sentences, subordinate and coordinated
  clauses, nominalizations, connective/subject openings, passive, first person, ';', ', not X', ...).
  Mahalanobis distance (Ledoit-Wolf) from the human mean, on pseudo-documents of 12 paragraphs.
  Longer texts are cut into chunks of 12 paragraphs; the median chunk distance is reported.
- Human thresholds (calib + test pooled): p95 = 5.50, p99 = 6.75. Test-only false alarms: 7.7% above p95, 1.7% above p99.
  Levels: typical <= p95 < borderline <= p99 < elevated <= 1.25 x p99 < high.
- Pseudo-documents above p99: Claude free 91%, human-sounding 95%, revise 98%, results/multipass 100%, polish 60%;
  other LLMs 68-93% (polish 12%).
- LD2 original 14.8 (high); LD2 after detector-guided revision 8.9 (high) while the classifier says AI 0%.
  So the revision lowered the classifier score, and only halfway the structure gap.
- Verdict rule: if the classifier says "likely human" but the distance is elevated/high, the verdict says so
  (possible detector-guided revision). Needs >= 6 paragraphs.
- Caveat: human pseudo-documents mix paragraphs from different papers, so paper-level variance is probably underestimated;
  treat "elevated" as a flag to inspect, not as proof.

## v4: adversarial training (2026-10-24)

Detector-guided rewrites as extra AI training data (`bench/generate_adv.py`). Each AI paragraph is rewritten by an LLM
(Claude sonnet/opus/haiku, five open-weight models) with two human paragraphs of train papers as style reference and the
rules that lower the detector score, scored by the running service and rewritten again until it passes (3 rounds in
the first run against v2+v3, 5 in the second against v2+v3+v4). Papers keep their split, so held-out papers have
held-out adversarial texts.

- Round 1 (`adv`): 1,698 paragraphs, 24% evaded v2+v3. Round 2 (`adv2`): 1,204 paragraphs, 7% evaded v2+v3+v4.
- `deberta-base-adv` = DeBERTa-v3-base trained on our data + RAID + adv + adv2 (2 epochs). v4 (adv only, 3 epochs, epoch 1
  kept) was an intermediate step and is not published.
- Held-out test papers, thresholds at alpha/k FPR of the calibration humans (`bench/eval_or.py`):

| ensemble | FPR body / abstract / RAID | Claude | open LLMs | adv | adv2 | polish | RAID |
|---|---|---|---|---|---|---|---|
| v2+v3 (v3 release) | .005 / .006 / .010 | .83 | .94 | .80 | .84 | .92 | .85 |
| v2+v3+v4 | .005 / .002 / .006 | .84 | .94 | .82 | .86 | .92 | .86 |
| v2+v3+v5 (intermediate) | .005 / .006 / .007 | .87 | .95 | .86 | .89 | .92 | .86 |
| v2+v3+v4+v5 | .003 / .002 / .006 | .86 | .95 | .85 | .89 | .91 | .87 |
| v5 alone | .002 / .004 / .009 | .74 | .85 | .86 | .90 | .58 | .85 |

  A model trained mostly on adversarial text alone loses plain Claude and polished text; the OR of the three keeps both.
- Document level (90 human papers): 1.1% flagged as "mixed" or worse, 0% as "likely AI".
- External test (LD2 manuscript, never used for training): original 40% of words AI (was 48% with v3 alone), 42% of
  paragraphs flagged by the v2+v3+v5 members at 1% FPR.

**Limit that remains.** LD2 after an agent-driven, paragraph-by-paragraph revision against the detector is flagged in
0% of paragraphs by every combination of v2-v5: our rewrites (short loops, one paragraph, stock rules) do not reach that
level of optimization, and a paragraph classifier cannot be assumed robust against an optimizer with feedback.
The document-level structural distance still flags it (8.9 vs human p99 6.75), which is why both signals are reported.
"Likely human" from the classifier alone is therefore not a proof of human authorship.


## v4.1: detector-guided manuscript revisions as training data (2026-10-24)

The paragraph classifiers v2-v5 did not see LD2 after its agent-driven revision (0% of paragraphs), and LD5 after its
revision only 2%. Both manuscripts are Claude-written and were revised paragraph by paragraph against this detector, so
their revised text is exactly the threat. Their original and revised paragraphs (`bench/data/ms_ld2.jsonl`,
`ms_ld5.jsonl`, 134 + 107 paragraphs, repeated 6x, with windows) are added as AI training rows.

**Does it generalize to an unseen manuscript?** Cross test, one manuscript in training, the other held out
(`deberta-base-xa`: LD2 only; `deberta-base-xb`: LD5 only; both with the full v5 data):

| model | held-out manuscript | paragraphs flagged before | after |
|---|---|---|---|
| xa (trained on LD2) | LD5 revised | 4% | 57% |
| xb (trained on LD5) | LD2 revised | 0% | 54% |
| v2+v3+xa | LD5 revised | 2% | 47% |
| v2+v3+xb | LD2 revised | 0% | 40% |

**Is it a false alarm on modern human text?** Control set `bench/data/human_2021.jsonl` (`bench/collect_human.py
--years 2021-2021`, `bench/eval_control.py`): 838 body paragraphs of 140 arXiv papers from 2021 (before ChatGPT, current
topics, never trained on). Paragraphs flagged: v2 0.7%, v3 0%, xa 0.5%, xb 0%, and 0.1-0.2% for every ensemble; no paper
has two flagged paragraphs.

**Final model.** `deberta-base-rev` = v5 data + both manuscripts. Ensemble v2 + v3 + rev (0.33% FPR each):

| held-out test papers | FPR body / abstract / RAID | Claude | open LLMs | adv | adv2 | polish | revise | RAID |
|---|---|---|---|---|---|---|---|---|
| v2+v3+rev | .005 / .004 / .006 | .94 | .96 | .93 | .96 | .91 | .95 | .89 |

- Control 2021: 0.1% of paragraphs flagged, 0 of 140 papers with >= 2 flagged. Document level on 90 human papers unchanged
  (1 "mixed", 0 "likely AI").
- LD2 original 99% AI words, LD2 revised 100%, LD5 revised 97%; LD1 revised 61% of paragraphs flagged.
  LD2 and LD5 are now in training, so these numbers are not a held-out test; the cross table above is. LD1 is
  an AI-assisted manuscript of the same author, so its flag rate cannot be called right or wrong.
- Remaining limit: only two manuscripts, one author and one revision procedure, define the "revised against a detector"
  class. A different optimizer (another prompt, model or number of rounds) can land elsewhere; the structural distance
  and a new round of rewrites against this ensemble are the countermeasures.

## Different optimizer, final ensemble (2026-10-25)

Test of robustness against a procedure the model was not trained on. `generate_adv.py` with `ADV_FEEDBACK=1`: after each
round the service returns the sentence scores, and the LLM is told to rewrite only the most flagged sentences (with
their scores), up to 6 rounds, stopping as soon as the paragraph is labelled `human` by the final ensemble v2+v3+rev.
Seeds are the AI paragraphs of earlier rounds (`ai.jsonl`, `ai_adv2.jsonl`), the rewriters are Claude (sonnet, opus, haiku)
and five open-weight models. These rewrites are not part of any training set.

- 323 paragraphs: 2 evaded (0.6%, 95% upper bound 2.2%); 316 (98%) still scored `ai` after the last round and 321 (99%)
  at least `uncertain`.
- By rewriter: gpt-oss 56/56 flagged, gemma3 51/51, mistral-small 52/54, phi4 41/42, qwen2.5 61/62, Claude 55/58 (the Claude
  calls are slow, so Claude is under-represented, 58 of 323). Both evasions came after 2-3 rounds; every other paragraph
  used all 6.
- For comparison, the earlier procedure (whole-paragraph rewrite, 5 rounds) evaded the previous ensemble in 7% of cases and
  the first ensemble in 24%.

This supports the claim for rewrites made by LLM loops with sentence-level feedback. It does not cover a human editing
the text by hand against the scores, nor a revision of a full manuscript with the whole paper as context, which is how
LD2 and LD5 were revised; those two are what the cross-manuscript test above measures (54-57% on an unseen manuscript).

## v4.2: third manuscript and a different optimizer (2026-10-25)

**Agent revision of an unseen manuscript.** LD4 (never in training; detector v4.1 = v2+v3+rev) was revised by
four agents section by section against the detector (procedure of the `ai-detect` skill, at most 5-6 rewrites per paragraph).
Paragraphs labelled: original 20 ai / 12 uncertain / 37 human of 69; revised 7 / 8 / 56 of 71. Words: AI 27% -> 9%.
Structural distance 17.9 -> 11.2 (human p99 6.75), so the document verdict became "likely human by the classifier, but
structure far from human papers". The revised text is flagged by the paragraph classifiers at 1% (v2+v3), and by
`deberta-base-rev` (trained on LD2 + LD5) at 21%. The agent loop therefore still reduced the score a lot,
and the generalization of `rev` to a third manuscript (21%) is weaker than between LD2 and LD5 (54-57%).

**Fix.** `deberta-base-rev2` adds the 66 paragraphs of LD4 that the agents rewrote or that were flagged and removed
(34 + 32), plus the 323 sentence-feedback rewrites of the held-out optimizer test. Ensemble v2 + v3 + rev2:

| held-out test papers | FPR body / abstract / RAID | Claude | open LLMs | adv | adv2 | polish | revise | RAID |
|---|---|---|---|---|---|---|---|---|
| v2+v3+rev (v4.1) | .005 / .004 / .006 | .94 | .96 | .93 | .96 | .91 | .95 | .89 |
| v2+v3+rev2 | .006 / .002 / .006 | .93 | .96 | .93 | .95 | .92 | .95 | .89 |

- Control 2021 (838 paragraphs of human papers on current topics): 0.2% flagged, no paper with two flagged paragraphs.
  Document level on 90 human papers unchanged.
- LD4, original and revised, are now flagged at 87% of paragraphs by v2+v3+rev2. LD4 is in training, so this is not a test,
  and it also hits the 37 original paragraphs that were not used. No pre-LLM paper by the same author could be found
  to separate "genuinely LLM-written" from "style of this author's papers". Treat flags on these manuscripts as
  strong evidence of an LLM-like style, not as a measure of authorship.
- A fourth manuscript (not in training) would be the next held-out test; none is reserved any more.

## Partially edited text (2026-10-25)

`bench/eval_hybrid.py`, held-out test papers, final ensemble v2+v3+rev2, paragraphs of at least 4 sentences, about 250 per row.
An LLM draft edited by a person is simulated by replacing a share of its sentences with human sentences of the same paper,
and the reverse by replacing a share of a human paragraph with AI sentences of the same paper.

| paragraph | share labelled `ai` | `ai` or `uncertain` |
|---|---|---|
| AI, all sentences kept | 99.5% | 99.5% |
| AI, 75% kept, rest human | 79.8% | 87.5% |
| AI, 50% kept, rest human | 30.8% | 45.2% |
| AI, 25% kept, rest human | 8.2% | 14.9% |
| human, untouched | 0.8% | 2.8% |
| human, 25% replaced by AI | 10.8% | 16.9% |
| human, 50% replaced by AI | 33.9% | 50.8% |

The detector responds roughly in proportion to the share of AI sentences. A draft in which about half of the sentences have
been rewritten by hand is flagged in a minority of cases, as expected for a paragraph classifier, and a human paragraph
with one sentence in four inserted by an LLM is flagged in about one case in ten. This is a simulation (sentence swaps from the
same paper), not real hand editing, which also changes the surviving sentences.

## Agent optimizer on unseen manuscripts (2026-10-25)

A different optimizer, never used for training: LLM agents revised the prose paragraph by paragraph with the final ensemble
(v2+v3+rev2) as the target, at most 5 rewrites per paragraph, plus a model-free structure check, then an independent review for
meaning. Manuscripts that `rev2` never saw: LD6, LD7, LD8, LD9, LD11 (LD2, LD4 and LD5 are in its training data and are left out).

| group | paragraphs | `human` before | `human` after | `ai` before → after |
|---|---|---|---|---|
| LD6, LD8, LD9 (mostly flagged at the start) | 175 → 174 | 15 (9%) | 16 (9%) | 154 → 138 |
| LD7, LD11 (about 40% already `human`) | 108 → 113 | 40 (37%) | 64 (57%) | 56 → 36 |
| all five | 283 → 287 | 55 (19%) | 80 (28%) | 210 → 174 |

For manuscripts the detector flags at the start, the optimizer with direct access to the target did not change the share of
paragraphs passing (9% before, 9% after) and moved some `ai` to `uncertain` (6 → 20). Where the paper started half
human-looking, more paragraphs passed (37% → 57%). The structural distance fell from high to typical or borderline in all five,
which the classifier does not use. These are agent loops with whole-section context, not a person; real hand editing is only
simulated (previous section).

## Leave-one-manuscript-out with two more training manuscripts: `rev3` (2026-10-25, not adopted)

`deberta-base-rev3` = `rev2` data plus the original and revised paragraphs of LD6 and LD8 (kept as local weights, not published).
LD9, LD7 and LD11 stay unseen. Ensemble v2 + v3 + member, paragraphs flagged at the 1%-FPR level:

| | LD9 original | LD9 revised | LD11 original | LD11 revised | LD7 original | LD7 revised | test FPR body/abs/RAID | Claude | adv |
|---|---|---|---|---|---|---|---|---|---|
| v2+v3+rev2 (published) | 83% | 69% | 56% | 24% | 48% | 40% | .006/.002/.006 | 93% | 93% |
| v2+v3+rev3 | 81% | 74% | 59% | 42% | 48% | 40% | .005/.002/.009 | 91% | 90% |

Control 2021 (838 human paragraphs): 0.2% flagged with rev2 in the ensemble, 0.1% with rev3.

Two more training manuscripts raise the detection of agent-revised text on unseen manuscripts by 5 points (LD9) and 18 points (LD11),
leave LD7 unchanged, keep the false positives at the same level, and lose 2 to 3 points on the in-distribution test. The gain is
small and the detection of a revised unseen manuscript stays at 74% for LD9, below the 90% I proposed as a target,
so `rev2` stays the published model. More manuscripts of this kind would probably help slowly; with one author and one revision
procedure, that is also where the risk of fitting the style of these papers grows.
