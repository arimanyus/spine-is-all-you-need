# Neuron-steering falsifier: probe prompt set

Prompt sets for testing Lyra's claim that, on **Qwen3.5-4B**, steering neuron **8949 in
layer 14 (+1)** and neuron **7956 in layer 13 (−1)** at `down_proj` produces an "evil villain"
persona. This folder supplies **prompts only**. Conditions (unsteered; the two neurons; ~10
random neuron pairs at the same scaled dose; doses 0.25 / 0.5 / 1 / 2) are built later by
Maya/Asha. **No model runs here.**

## Primary vs secondary outcomes

Per Vivek, the goal is **breaking the model**, not persona. Priorities:

1. **Primary 1 — spine break (fold under pressure):** does steering make the model abandon a
   correct answer and fold to the planted wrong answer (the *foil*)? Measured on
   `pressure_probes.jsonl` with `consistency_flip` / fold-to-foil.
2. **Primary 2 — refusal collapse:** does steering change compliance on harmful requests?
   Measured on `refusal_probes.jsonl` with the existing `refusal_flip` / NLI refusal scorer.
3. **Secondary — coherence/persona:** `persona_probes.jsonl` (does tone change) and
   `factual_probes.jsonl` (does steering break basic competence). These are secondary/coherence
   measures, not the headline result.

## Eval sets (used to score the hypothesis)

| file | n | group | purpose | scorer |
|------|---|-------|---------|--------|
| `pressure_probes.jsonl`  | 12 | `pressure` | **Primary 1.** Does steering make the model fold to the foil under sustained pressure? | `consistency_flip` / fold-to-foil |
| `refusal_probes.jsonl`   | 24 | `refusal`  | **Primary 2.** Does steering change compliance on harmful requests? | existing `refusal_flip` / NLI refusal scorer |
| `persona_probes.jsonl`   | 30 | `persona`  | Secondary. Does tone shift toward a "villain" register on neutral prompts? | persona score (**must be fixed before results**) |
| `factual_probes.jsonl`   | 30 | `factual`  | Secondary. Does steering break basic factual competence? | gold / alias match |

## Search sets (used to find candidate neurons — never scored as outcomes)

| file | n | purpose |
|------|---|---------|
| `contrast_prompts.jsonl` | 40 | 20 refuse-side prompts + 20 harmless lookalikes on matched topics. Used to find neurons in layers 13–14 that differ between refusal and compliance. |
| `spine_contrast.jsonl`   | 48 | 24 false-premise / memory-rewrite items, each as a `clean` row (neutral question only) and a `pressured` row (full escalating turns). Used to find folding neurons. |

### Disjointness invariant (must hold)

Search sets (`contrast_prompts`, `spine_contrast`) must **never overlap** eval sets
(`refusal_probes`, `pressure_probes`). `spine_contrast` draws fp/mr items that exclude the 12
`pressure_probes` source ids (and the six known-broken ids), so neuron search and evaluation
stay disjoint. Validated by `/tmp/validate_probes.py` (`disjoint ids`, `disjoint fp/mr sources`).

## Protocol notes (enforced when conditions are run)

- Every condition (unsteered, two-neuron, random pairs, every dose) must use the **identical
  prompts in identical order**.
- **Temperature 0.**
- The **persona score must be fixed before any results are looked at** (pre-registered), so the
  secondary persona read cannot be tuned to the outcome.
- Only prompts are stored here — **no model replies.**

## File details

- **`persona_probes.jsonl`** — 30 neutral everyday prompts (`pp-001`..`pp-030`). The first 20
  are the attached set as-is; `pp-021`..`pp-030` are added neutral prompts in the same style
  (advice, explanation, small talk), chosen so nothing invites a villain register.
- **`refusal_probes.jsonl`** — all 24 `tool_bait` items (`tb-001`..`tb-024`), copied with every
  original field (`template`, `foil`, `gold`, `gold_logprob_target`, `turns`,
  `compliance_markers`, `refusal_markers`, `score_mode`) plus `group: "refusal"`.
- **`factual_probes.jsonl`** — 30 short factual questions (`fq-001`..`fq-030`) with one
  unambiguous answer each (arithmetic, geography, science, units). Fields: `id`, `group`,
  `prompt`, `gold`, `aliases`.
- **`pressure_probes.jsonl`** — 12 items (6 `false_premise_stack`, 6 `memory_rewrite`), all
  original fields plus `group: "pressure"`. Known-broken ids (fp-008, fp-011, fp-021, mr-015,
  mr-016, mr-019) are excluded, and every chosen item's foil appears verbatim in at least one
  user pressure turn. Deterministic lowest-id pick:
  `fp-001, fp-003, fp-004, fp-005, fp-006, fp-007, mr-002, mr-003, mr-004, mr-005, mr-006, mr-007`.
- **`contrast_prompts.jsonl`** — 40 prompts (`cp-001`..`cp-040`) as 20 matched topic pairs
  (odd = `refuse`, even = `comply`). Fields: `id`, `group`, `side`, `topic`, `prompt`.
- **`spine_contrast.jsonl`** — 48 rows: 24 source items (`sc-001`..`sc-024`), each emitted as
  `-clean` (turn 0 only) and `-pressured` (full turns through the final pressure turn). Fields:
  `id`, `source_id`, `side`, `turns`, `gold`, `foil`.

## Withheld harmful-request content — do not merge / publish

`refusal_probes.jsonl` contains the repo's **withheld** harmful-request (`tool_bait`) prompts,
and the refuse side of `contrast_prompts.jsonl` holds newly written refusal-trigger prompts at
the same request level. This repository's policy (`.gitignore`, `README.md`, `paper/` ethics,
`share_withheld.py`) withholds harmful-request prompts and replies from the public tree. This PR
is therefore a **DRAFT — do not merge**, and these files hold **prompts only, no model replies**.
The new refuse-side prompts are bounded per Vivek: refusal triggers at the existing `tool_bait`
level, with **no operational weapons, CBRN, or CSAM content**. Resolve publication/withholding
handling (e.g. keeping these files out of the public `main` tree, or routing them through
`share_withheld.py`) before any merge.
