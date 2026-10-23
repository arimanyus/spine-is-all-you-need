# Order Is All You Need: locked protocol

Do small open LLM judges change their pairwise verdict when the two answers swap slots? This
sub-project measures position (order) bias of small open judges and whether it is large enough to
reshuffle a model ranking. This folder holds the scorer only; the pair sets (Kenji) and the judge
runner (Maya) live elsewhere and feed it JSONL.

## Scope caveat

Position bias in LLM judges is already published: Wang et al. 2023, *Large Language Models are not
Fair Evaluators* ([arXiv 2305.17926](https://arxiv.org/abs/2305.17926)), and Zheng et al. 2023,
*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* ([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)).
What is new here is narrow: (1) small open judges that fit on a CPU, (2) whether the bias is big
enough to reshuffle a model ranking rather than individual verdicts, and (3) memorization masking,
tested by comparing a possibly memorized pair set against a fresh one. Numbers from these judges are
seeds, not frontier claims; nothing here says anything about frontier judges until the same protocol
is run against them.

## Protocol (locked; do not change thresholds or prompt)

1. Every pair is judged twice: order **AB** (answer1 in slot A, answer2 in slot B) and order **BA**
   (answer2 in slot A, answer1 in slot B).
2. The judge outputs a single token: `A`, `B`, or `C` (tie). Greedy decoding.
3. Fixed prompt, in `order/prompt.py` as `SYSTEM` and `USER`:

   System: `You are an impartial judge. Compare two responses to the user's question. Judge
   correctness first, then helpfulness. Do not let response order or length influence you.`

   User: `[Question]\n{question}\n\n[Response A]\n{a}\n\n[Response B]\n{b}\n\nOutput only one
   token: A, B, or C (tie).`

4. Two pair sets, reported separately:
   * **Set M** (~80 pairs): MT-Bench human judgments, 6 models, strata by human margin
     `clear` / `slight` / `tie`. Possibly memorized by the judges.
   * **Set F** (~40 pairs): fresh. 10 identical-answer controls (`stratum = identical`,
     answer1 == answer2) + 30 length-matched near-ties (`stratum = near_tie`).

## Pre-registered checks

| check | holds / fires when |
|---|---|
| per-verdict null | full-flip rate < 10% with Wilson 95% upper bound < 15%, **and** slot-A preference on identical-answer controls within [0.40, 0.60] |
| ranking null | Kendall tau between the AB-order and BA-order model rankings >= 0.8 |
| leakage signal | set F flip rate > set M flip rate with non-overlapping Wilson 95% CIs |

The scorer also reports the per-verdict check with the half-flip-inclusive rate
(`pass_half_inclusive`) as a secondary, stricter reading. A check is `null` (not evaluable) when
its inputs are missing: no identical controls, fewer than two scorable models, or a set absent.

## Scorer (`order/scorer.py`)

```
python -m order.scorer --pairs pairs.jsonl --judgments judgments.jsonl --out results/
python -m order.scorer ... --rank-method bt            # Bradley-Terry instead of win rate
python -m order.scorer ... --field human_winner=gold   # rename any input field
```

Writes `summary.json` and `summary.md` (per judge, per set, per stratum, plus the checks).

Metrics, in protocol order:

1. **Verdict parsing** `parse_verdict(raw)` -> `A` / `B` / `C` / `UNPARSEABLE`. Accepts a bare
   letter with any wrapping (`**A**`, `"B"`, `Answer: B.`, `C (tie)`, `Response A.`, `tie`), a
   leading letter followed by an explanation, and longer texts that name exactly one of A/B
   (`Response B is more accurate`, `I prefer A`, `the winner is B`) or use a tie word without
   singling out a letter. Empty output, `A or B`, and anything else is `UNPARSEABLE`. Because the
   prompt demands one leading token, a leading `A` is read as the verdict even in prose such as
   `A good answer...`; check `unparseable` and spot-check raw outputs per judge.
   Unparseable is its own category: counted and reported (rate with CI over all judgments), never
   dropped. A missing judgment row is counted as unparseable (`missing_judgments`).
2. **Mapping** `slot_to_answer(verdict, order)` -> `"1"` / `"2"` / `"tie"` / `"unparseable"`.
3. **Flip** per pair, from the two answer-identity verdicts: 0 if equal, 1 if the winner differs
   (full flip), 0.5 if tie in one order and a win in the other (tie mismatch). Pairs with an
   unparseable order are excluded from the flip denominator (`pairs_scored_for_flip`). Reported:
   `full_flip` (score == 1), `any_flip` (score > 0, half-inclusive), `mean_flip_score`,
   `tie_mismatch` count.
4. **Slot-A preference** = #A / (#A + #B) over all judgments of the block (ties excluded from the
   denominator; `tie_rate` is reported alongside). `identical_controls` is this block restricted
   to pairs with `stratum == identical` or `answer1 == answer2`.
5. **Accuracy vs human_winner** per order, over pairs with a non-null `human_winner`; a tie
   verdict matches a human tie; unparseable counts as a miss.
6. **Delta-logprob**: with `d = logP(A) - logP(B)` at the verdict position, the answer-identity
   margin for answer1 is `+d` in AB and `-d` in BA. The order-induced shift per pair is
   `d_AB + d_BA` (positive = leans towards whatever sits in slot A); reported as mean/median plus
   the mean slot margin per order. Only pairs with log-probs in both orders count (`n`).
7. **Ranking**: per order, model win rate from pairwise outcomes (tie = half a win; Bradley-Terry
   via `--rank-method bt`), models sorted descending; Kendall tau-b between the AB and BA scores
   over the common models. `null` when fewer than two models are scorable.
8. **Wilson 95% CIs** on every rate: `{"k", "n", "rate", "ci_lo", "ci_hi"}`; `rate` is `null`
   and the CI is [0, 1] when `n == 0`.
9. **Checks** as above, per judge, computed on the pooled set (`all`) with the identical-control
   slot-A preference and the F-vs-M flip comparison.

## Input schema

Field names below are the defaults; all are configurable via `Fields` / `--field key=name`.

### pairs.jsonl (Kenji) -- one row per pair

| field | type | notes |
|---|---|---|
| `id` | string | unique pair id, joined to judgments |
| `set` | `"M"` or `"F"` | reported separately |
| `stratum` | string | set M: `clear` / `slight` / `tie`; set F: `identical` / `near_tie` |
| `question` | string | shown to the judge |
| `answer1`, `answer2` | string | answer1 is slot A in order AB, slot B in order BA |
| `model1`, `model2` | string or null | producers of answer1/answer2; needed for rankings (set M), null is fine on set F |
| `human_winner` | `1`, `2`, `"tie"`, or null | ground truth for accuracy; null on set F |

### judgments.jsonl (Maya) -- one row per (judge, pair, order)

| field | type | notes |
|---|---|---|
| `pair_id` | string | matches `pairs.id` |
| `judge` | string | judge model name; the report is grouped by it |
| `order` | `"AB"` or `"BA"` | which answer sat in slot A (see `order/prompt.py::slot_answers`) |
| `raw` | string | the judge's raw text output (not pre-parsed) |
| `logprob_a`, `logprob_b` | float or null | log-probs of the tokens `A` and `B` at the verdict (first generated) position; optional |

Exactly one row per (judge, pair, order); duplicates are rejected. Build the prompt with
`order.prompt.build_messages(question, *slot_answers(pair, order))` so both orders use the locked
text verbatim.

## Tests

`pytest -q tests/test_order_scorer.py` runs on CPU with synthetic fixtures, no model weights.
