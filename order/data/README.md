# Order Is All You Need: pair sets

Input data for the position-bias scorer (`order/scorer.py`, PR #18). One row per pair,
JSONL, schema as in `order/PROTOCOL.md`:

| field | values |
|---|---|
| `id` | unique pair id, joined to `judgments.jsonl` |
| `set` | `"F"` or `"M"` |
| `stratum` | set F: `near_tie` / `identical`; set M: `clear` / `slight` / `tie` |
| `question`, `answer1`, `answer2` | strings shown to the judge |
| `model1`, `model2` | producers of `answer1` / `answer2` |
| `human_winner` | `1`, `2`, or `"tie"` |

Both files are produced by the deterministic scripts beside them (`python build_set_f.py`,
`python build_set_m.py`, written to the current directory).

## Set F: `set_f.jsonl` (40 rows, fresh)

Freshly handwritten for this study, so no judge could have trained on it. `model1` and
`model2` are the placeholder `fresh-handwritten`; set F is not used for model rankings.

- `near_tie` (30): two handwritten answers of equivalent quality and content, length-matched
  within 10% (max ratio 1.096). `human_winner: "tie"` is the author's label, not a human vote.
- `identical` (10): the same answer in both slots (`answer1 == answer2`). Pure
  position-bias control: any non-tie verdict is a slot preference.

## Set M: `set_m.jsonl` (80 rows, MT-Bench)

First-turn pairs from `lmsys/mt_bench_human_judgments` (split `human`), stratified by human
agreement among the judges who saw the same (question, model pair):

- `clear` (27): at least 2 judges, all unanimous on a winner.
- `slight` (27): a majority winner, but not unanimous.
- `tie` (26): a majority of judges voted tie.
- Pairs with a split vote (no majority) are dropped.

Pairs where either answer exceeds 1500 characters are excluded. Strata are sampled with
`random.seed(1337)`. Each row also carries `question_id` (MT-Bench) and
`possibly_memorized: true`: these are public benchmark answers that the judges may have
seen in training, which is the point of the F-vs-M leakage comparison in the protocol.

Note: three `tie` rows (`m-tie-002`, `m-tie-011`, `m-tie-018`) have byte-identical answers
from gpt-3.5-turbo and gpt-4 (short factual outputs). The scorer treats `answer1 == answer2`
as an identical control, so they are pooled into its `identical_controls` block as well.

## Schema change from the raw builder output

`set_m.jsonl` originally encoded `human_winner` as `"answer1"` / `"answer2"`. The scorer's
`normalize_human` accepts only `1`, `2`, `"tie"` (or `0` / `"draw"` / null) and raises on
anything else, so the data and `build_set_m.py` now emit `1` / `2`. No other field changed.
