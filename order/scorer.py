"""Scorer for the order-bias protocol (see order/PROTOCOL.md). Pure Python, no model code.

Inputs are two JSONL files joined on the pair id:

* pairs      -- one row per pair: id, set (M|F), stratum, question, answer1, answer2,
                model1, model2, human_winner (1|2|tie|null). Built by Kenji.
* judgments  -- one row per (judge, pair, order): pair_id, judge, order (AB|BA), raw
                (judge's text), logprob_a / logprob_b (log-probs of the tokens "A" and "B" at
                the verdict position, optional). Produced by Maya's runner.

Field names are configurable through `Fields`. Verdicts are parsed to A/B/C/UNPARSEABLE and
then mapped to answer identity "1"/"2"/"tie"/"unparseable" per order. Unparseable judgments are
never dropped silently: they are counted and reported, and excluded only from the denominators
whose definition requires a verdict (flip, slot-A preference, ranking).

CLI: python -m order.scorer --pairs pairs.jsonl --judgments judgments.jsonl --out results/
"""

import argparse
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, fields as dc_fields

UNPARSEABLE = "UNPARSEABLE"
TIE = "tie"
Z95 = 1.959963984540054

# Pre-registered thresholds (locked; see PROTOCOL.md).
FLIP_RATE_MAX = 0.10
FLIP_CI_HI_MAX = 0.15
SLOT_A_IDENTICAL_RANGE = (0.40, 0.60)
KENDALL_TAU_MIN = 0.8


@dataclass(frozen=True)
class Fields:
    """JSON keys in the pairs and judgments files. Override any of them (CLI: --field key=name)."""

    # pairs file
    id: str = "id"
    set: str = "set"
    stratum: str = "stratum"
    answer1: str = "answer1"
    answer2: str = "answer2"
    model1: str = "model1"
    model2: str = "model2"
    human_winner: str = "human_winner"
    # judgments file
    pair_id: str = "pair_id"
    judge: str = "judge"
    order: str = "order"
    raw: str = "raw"
    logprob_a: str = "logprob_a"
    logprob_b: str = "logprob_b"
    # stratum name of the identical-answer controls (set F)
    identical_stratum: str = "identical"


DEFAULT_FIELDS = Fields()


# ----------------------------------------------------------------------------- 1. parsing

_SLOT = r"(?:response|answer|option)\s+"
_LABEL = r"(?:final\s+)?(?:verdict|answer|output|choice|winner|response|option|judg(?:e)?ment|preference)"
_STRIP = re.compile(r"^[\s\W_]*(?:" + _LABEL + r"\s*[:=\-]\s*)?[\s\W_]*", re.IGNORECASE)
# Leading letter, optionally prefixed by "Response"; not "A or B" / "A tie".
_LONE = re.compile(r"^(?:" + _SLOT + r")?([ABC])\b(?!\s*(?:and|or|vs\.?|/)\s*(?:" + _SLOT + r")?[ABC]\b)(?!\s+tie\b)",
                   re.IGNORECASE)
_BOTH = re.compile(r"\b[AB]\b\s*(?:and|or|vs\.?|/)\s*(?:" + _SLOT + r")?[AB]\b", re.IGNORECASE)
_TIE_WORD = re.compile(r"\b(?:tie|tied|draw|equal(?:ly)?|both|neither|same)\b", re.IGNORECASE)
_MENTIONS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b" + _SLOT + r"([AB])\b",
    r"\b(?:answer|response|option|verdict|winner|choice)\s+(?:is|:)\s*([ABC])\b",
    r"\b(?:prefer|choose|pick|select|go with)\s+(?:" + _SLOT + r")?([ABC])\b",
    r"\b([AB])\s+is\s+(?:better|correct|more|superior|the\b)",
)]


def parse_verdict(raw):
    """Map raw judge text to 'A', 'B', 'C' or UNPARSEABLE.

    Accepts a bare letter with any wrapping (whitespace, quotes, markdown, 'Answer: B', 'C (tie)',
    'Response A.', 'tie'), and a leading letter followed by an explanation. Longer texts count only
    if they name exactly one of A/B ('Response B is ...', 'I prefer A', 'the winner is B'), or use a
    tie word ('both', 'equal', 'tie') without singling out a letter. 'A or B', 'A and B' without a
    tie word, empty output and anything else is UNPARSEABLE.
    """
    if raw is None:
        return UNPARSEABLE
    text = _STRIP.sub("", str(raw).strip(), count=1)
    if not text:
        return UNPARSEABLE
    m = _LONE.match(text)
    if m:
        return m.group(1).upper()
    tie = bool(_TIE_WORD.search(text))
    if _BOTH.search(text):
        return "C" if tie else UNPARSEABLE
    letters = {m.upper() for rx in _MENTIONS for m in rx.findall(text)}
    if len(letters) == 1:
        return letters.pop()
    if not letters and tie:
        return "C"
    return UNPARSEABLE


# ----------------------------------------------------------------------------- 2. mapping

def slot_to_answer(verdict, order):
    """Slot verdict (A/B/C/UNPARSEABLE) -> answer identity '1' / '2' / 'tie' / 'unparseable'."""
    if order not in ("AB", "BA"):
        raise ValueError(f"order must be AB or BA, got {order!r}")
    if verdict == "C":
        return TIE
    if verdict == "A":
        return "1" if order == "AB" else "2"
    if verdict == "B":
        return "2" if order == "AB" else "1"
    return "unparseable"


def normalize_human(value):
    """human_winner field -> '1' / '2' / 'tie' / None."""
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in ("1", "2"):
        return s
    if s in ("tie", "0", "draw"):
        return TIE
    raise ValueError(f"unrecognised human_winner {value!r}")


# ----------------------------------------------------------------------------- 3. flips

def flip_score(w_ab, w_ba):
    """0 = same answer-identity verdict in both orders; 1 = winner differs; 0.5 = tie in one order,
    win in the other (a 'tie mismatch'). None if either order is unparseable."""
    if "unparseable" in (w_ab, w_ba):
        return None
    if w_ab == w_ba:
        return 0.0
    if TIE in (w_ab, w_ba):
        return 0.5
    return 1.0


# ----------------------------------------------------------------------------- 8. Wilson CI

def wilson(k, n, z=Z95):
    """Wilson score interval (lo, hi) for k successes in n trials. (0, 1) when n == 0."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k, n):
    """{'k', 'n', 'rate', 'ci_lo', 'ci_hi'}; rate is None when n == 0."""
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci_lo": lo, "ci_hi": hi}


# ----------------------------------------------------------------------------- 7. ranking

def win_rates(outcomes):
    """outcomes: iterable of (model1, model2, winner in '1'/'2'/'tie'). Returns {model: win rate},
    ties counting half. Unparseable outcomes must be filtered out by the caller."""
    wins, games = defaultdict(float), defaultdict(int)
    for m1, m2, w in outcomes:
        games[m1] += 1
        games[m2] += 1
        if w == TIE:
            wins[m1] += 0.5
            wins[m2] += 0.5
        else:
            wins[m1 if w == "1" else m2] += 1
    return {m: wins[m] / games[m] for m in games}


def bradley_terry(outcomes, iters=200):
    """Bradley-Terry strengths via the MM algorithm (Hunter 2004), ties as half wins, normalised
    to sum to 1. Optional alternative to `win_rates`."""
    wins, n_ij = defaultdict(float), defaultdict(int)
    for m1, m2, w in outcomes:
        n_ij[frozenset((m1, m2))] += 1
        if w == TIE:
            wins[m1] += 0.5
            wins[m2] += 0.5
        else:
            wins[m1 if w == "1" else m2] += 1
    models = sorted({m for pair in n_ij for m in pair})
    p = {m: 1.0 for m in models}
    for _ in range(iters):
        new = {}
        for m in models:
            denom = sum(n / (p[m] + p[o]) for pair, n in n_ij.items() if m in pair for o in pair - {m})
            new[m] = wins[m] / denom if denom else 0.0
        total = sum(new.values()) or 1.0
        p = {m: v / total for m, v in new.items()}
    return p


def rank(scores):
    """Models sorted by descending score (ties broken by name for determinism)."""
    return sorted(scores, key=lambda m: (-scores[m], m))


def kendall_tau(x, y):
    """Kendall tau-b between two equal-length score sequences (ties handled). None if undefined."""
    n = len(x)
    if n != len(y):
        raise ValueError("sequences must have equal length")
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                tx += 1
            if dy == 0:
                ty += 1
            if dx * dy > 0:
                conc += 1
            elif dx * dy < 0:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    return (conc - disc) / denom if denom else None


def ranking_summary(pairs, verdicts, method="winrate"):
    """Per-order model rankings and Kendall tau between them. `verdicts[(pair_id, order)]` is the
    answer-identity verdict. Returns None when fewer than two models are scorable."""
    scorer = bradley_terry if method == "bt" else win_rates
    scores = {}
    for order in ("AB", "BA"):
        outcomes = [
            (p["model1"], p["model2"], verdicts[(p["id"], order)])
            for p in pairs
            if p["model1"] is not None and p["model2"] is not None
            and verdicts.get((p["id"], order), "unparseable") != "unparseable"
        ]
        scores[order] = scorer(outcomes)
    models = sorted(set(scores["AB"]) & set(scores["BA"]))
    if len(models) < 2:
        return None
    tau = kendall_tau([scores["AB"][m] for m in models], [scores["BA"][m] for m in models])
    return {
        "method": method,
        "scores": scores,
        "ranking": {o: rank(scores[o]) for o in scores},
        "kendall_tau": tau,
    }


# ----------------------------------------------------------------------------- loading

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(rows, fields=DEFAULT_FIELDS):
    """Rows (dicts with the configured keys) -> canonical pair dicts keyed by the canonical names."""
    out = []
    for r in rows:
        out.append({
            "id": str(r[fields.id]),
            "set": r.get(fields.set),
            "stratum": r.get(fields.stratum),
            "model1": r.get(fields.model1),
            "model2": r.get(fields.model2),
            "human_winner": normalize_human(r.get(fields.human_winner)),
            "identical": r.get(fields.stratum) == fields.identical_stratum
            or (fields.answer1 in r and r.get(fields.answer1) == r.get(fields.answer2)),
        })
    return out


def load_judgments(rows, fields=DEFAULT_FIELDS):
    """Rows -> {judge: {(pair_id, order): judgment}} with parsed slot verdicts and log-probs."""
    by_judge = defaultdict(dict)
    for r in rows:
        key = (str(r[fields.pair_id]), r[fields.order])
        if key in by_judge[r[fields.judge]]:
            raise ValueError(f"duplicate judgment for judge={r[fields.judge]} pair={key[0]} order={key[1]}")
        by_judge[r[fields.judge]][key] = {
            "verdict": parse_verdict(r.get(fields.raw)),
            "logprob_a": r.get(fields.logprob_a),
            "logprob_b": r.get(fields.logprob_b),
        }
    return by_judge


# ----------------------------------------------------------------------------- summary

def _delta(j):
    a, b = j["logprob_a"], j["logprob_b"]
    return None if a is None or b is None else a - b


def summarize_pairs(pairs, judgments, rank_method="winrate"):
    """Full summary block for one judge over `pairs`. `judgments` maps (pair_id, order) -> judgment.
    Pairs missing a judgment for an order are counted in `missing_judgments` and treated as
    unparseable for that order."""
    n_j = n_unp = 0
    missing = 0
    slot_a = slot_b = slot_c = 0
    flips, tie_mismatch = [], 0
    answers = {}
    acc = {"AB": [0, 0], "BA": [0, 0]}
    shifts, slot_margin = [], {"AB": [], "BA": []}
    for p in pairs:
        per_order = {}
        for order in ("AB", "BA"):
            j = judgments.get((p["id"], order))
            if j is None:
                missing += 1
                j = {"verdict": UNPARSEABLE, "logprob_a": None, "logprob_b": None}
            n_j += 1
            v = j["verdict"]
            n_unp += v == UNPARSEABLE
            slot_a += v == "A"
            slot_b += v == "B"
            slot_c += v == "C"
            w = slot_to_answer(v, order)
            answers[(p["id"], order)] = w
            per_order[order] = (w, _delta(j))
            if p["human_winner"] is not None:
                acc[order][1] += 1
                acc[order][0] += w == p["human_winner"]
        (w_ab, d_ab), (w_ba, d_ba) = per_order["AB"], per_order["BA"]
        f = flip_score(w_ab, w_ba)
        if f is not None:
            flips.append(f)
            tie_mismatch += f == 0.5
        if d_ab is not None and d_ba is not None:
            # Answer-identity margin favouring answer1: +d in AB, -d in BA. Their difference is the
            # order-induced shift; positive means the judge leans towards whatever sits in slot A.
            shifts.append(d_ab - (-d_ba))
            slot_margin["AB"].append(d_ab)
            slot_margin["BA"].append(d_ba)

    n_flip = len(flips)
    decisive = slot_a + slot_b
    out = {
        "n_pairs": len(pairs),
        "n_judgments": n_j,
        "missing_judgments": missing,
        "unparseable": rate(n_unp, n_j),
        "pairs_scored_for_flip": n_flip,
        "full_flip": rate(sum(f == 1.0 for f in flips), n_flip),
        "any_flip": rate(sum(f > 0 for f in flips), n_flip),
        "mean_flip_score": sum(flips) / n_flip if n_flip else None,
        "tie_mismatch": tie_mismatch,
        "slot_a_pref": rate(slot_a, decisive),
        "tie_rate": rate(slot_c, n_j - n_unp),
        "accuracy": {o: rate(*acc[o]) for o in acc},
        "delta_logprob": {
            "n": len(shifts),
            "shift_mean": statistics.fmean(shifts) if shifts else None,
            "shift_median": statistics.median(shifts) if shifts else None,
            "slot_margin_mean": {o: statistics.fmean(v) if v else None for o, v in slot_margin.items()},
        },
        "ranking": ranking_summary(pairs, answers, rank_method),
    }
    return out


def _flip_ok(r):
    return r["rate"] is not None and r["rate"] < FLIP_RATE_MAX and r["ci_hi"] < FLIP_CI_HI_MAX


def checks(all_, identical, set_f, set_m):
    """Pre-registered checks for one judge from its summary blocks: pooled (`all_`), identical-answer
    controls, set F and set M (the last three may be None). pass/signal is None when not evaluable
    (no identical controls, fewer than two models, a set missing)."""
    flip, any_flip = all_["full_flip"], all_["any_flip"]
    slot = identical["slot_a_pref"] if identical else None
    lo, hi = SLOT_A_IDENTICAL_RANGE
    slot_ok = None if slot is None or slot["rate"] is None else lo <= slot["rate"] <= hi

    def verdict_pass(r):
        return None if slot_ok is None or r["rate"] is None else bool(_flip_ok(r) and slot_ok)

    tau = (all_["ranking"] or {}).get("kendall_tau")

    f_flip = set_f["full_flip"] if set_f else None
    m_flip = set_m["full_flip"] if set_m else None
    leak = None
    if f_flip and m_flip and f_flip["rate"] is not None and m_flip["rate"] is not None:
        leak = f_flip["rate"] > m_flip["rate"] and f_flip["ci_lo"] > m_flip["ci_hi"]

    return {
        "per_verdict_null": {
            "pass": verdict_pass(flip),
            "pass_half_inclusive": verdict_pass(any_flip),
            "full_flip_rate": flip["rate"],
            "full_flip_ci_hi": flip["ci_hi"],
            "any_flip_rate": any_flip["rate"],
            "any_flip_ci_hi": any_flip["ci_hi"],
            "slot_a_pref_identical": None if slot is None else slot["rate"],
            "slot_a_pref_identical_n": None if slot is None else slot["n"],
        },
        "ranking_null": {"pass": None if tau is None else bool(tau >= KENDALL_TAU_MIN), "kendall_tau": tau},
        "leakage_signal": {
            "signal": leak,
            "flip_F": None if f_flip is None else f_flip["rate"],
            "flip_M": None if m_flip is None else m_flip["rate"],
        },
    }


def summarize(pairs, by_judge, rank_method="winrate"):
    """Per-judge, per-set, per-stratum summary plus pre-registered checks."""
    sets = sorted({p["set"] for p in pairs}, key=str)
    report = {"n_pairs": len(pairs), "sets": sets, "judges": {}}
    for judge, judgments in sorted(by_judge.items()):
        js = {"all": summarize_pairs(pairs, judgments, rank_method)}
        for s in sets:
            sp = [p for p in pairs if p["set"] == s]
            block = summarize_pairs(sp, judgments, rank_method)
            block["strata"] = {
                st: summarize_pairs([p for p in sp if p["stratum"] == st], judgments, rank_method)
                for st in sorted({p["stratum"] for p in sp}, key=str)
            }
            js[s] = block
        identical_pairs = [p for p in pairs if p["identical"]]
        identical = summarize_pairs(identical_pairs, judgments, rank_method) if identical_pairs else None
        report["judges"][judge] = {
            "sets": js,
            "identical_controls": identical,
            "checks": checks(js["all"], identical, js.get("F"), js.get("M")),
        }
    return report


# ----------------------------------------------------------------------------- markdown

def _pct(r):
    if r is None or r["rate"] is None:
        return "-"
    return f"{100 * r['rate']:.1f}% [{100 * r['ci_lo']:.1f}, {100 * r['ci_hi']:.1f}] (n={r['n']})"


def _num(x, fmt="{:+.3f}"):
    return "-" if x is None else fmt.format(x)


def _flag(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def to_markdown(report):
    head = ("| judge | set | stratum | pairs | full flip | any flip (half incl.) | tie mismatch | slot-A pref | "
            "unparseable | acc AB | acc BA | dlogprob shift mean / median | Kendall tau |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for judge, jr in report["judges"].items():
        for set_name, block in jr["sets"].items():
            items = [("(all)", block)] + [(st, b) for st, b in block.get("strata", {}).items()]
            for stratum, b in items:
                dl, rk = b["delta_logprob"], b["ranking"]
                rows.append(
                    f"| {judge} | {set_name} | {stratum} | {b['n_pairs']} | {_pct(b['full_flip'])} | "
                    f"{_pct(b['any_flip'])} | {b['tie_mismatch']} | {_pct(b['slot_a_pref'])} | "
                    f"{_pct(b['unparseable'])} | {_pct(b['accuracy']['AB'])} | {_pct(b['accuracy']['BA'])} | "
                    f"{_num(dl['shift_mean'])} / {_num(dl['shift_median'])} (n={dl['n']}) | "
                    f"{_num(None if rk is None else rk['kendall_tau'], '{:.3f}')} |"
                )
    check_head = ("\n| judge | per-verdict null (full flip) | per-verdict null (half incl.) | ranking null (tau >= 0.8) | "
                  "leakage signal (F > M, disjoint CIs) |\n|---|---|---|---|---|\n")
    check_rows = []
    for judge, jr in report["judges"].items():
        c = jr["checks"]
        pv, rn, lk = c["per_verdict_null"], c["ranking_null"], c["leakage_signal"]
        check_rows.append(
            f"| {judge} | {_flag(pv['pass'])} (flip {_num(pv['full_flip_rate'], '{:.3f}')}, "
            f"CI hi {_num(pv['full_flip_ci_hi'], '{:.3f}')}, slot-A ident {_num(pv['slot_a_pref_identical'], '{:.3f}')}) | "
            f"{_flag(pv['pass_half_inclusive'])} | {_flag(rn['pass'])} (tau {_num(rn['kendall_tau'], '{:.3f}')}) | "
            f"{'n/a' if lk['signal'] is None else ('YES' if lk['signal'] else 'no')} "
            f"(F {_num(lk['flip_F'], '{:.3f}')}, M {_num(lk['flip_M'], '{:.3f}')}) |"
        )
    return ("# Order-bias summary\n\nRates are shown with Wilson 95% CIs.\n\n" + head + "\n".join(rows)
            + "\n\n## Pre-registered checks\n" + check_head + "\n".join(check_rows) + "\n")


# ----------------------------------------------------------------------------- CLI

def score_files(pairs_path, judgments_path, fields=DEFAULT_FIELDS, rank_method="winrate"):
    pairs = load_pairs(load_jsonl(pairs_path), fields)
    by_judge = load_judgments(load_jsonl(judgments_path), fields)
    report = summarize(pairs, by_judge, rank_method)
    report["fields"] = asdict(fields)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out", required=True, help="output directory (summary.json, summary.md)")
    ap.add_argument("--rank-method", choices=["winrate", "bt"], default="winrate")
    ap.add_argument("--field", action="append", default=[], metavar="KEY=NAME",
                    help="rename an input field, e.g. --field human_winner=gold_winner "
                         f"(keys: {', '.join(f.name for f in dc_fields(Fields))})")
    args = ap.parse_args(argv)
    overrides = dict(kv.split("=", 1) for kv in args.field)
    fields = Fields(**overrides)
    report = score_files(args.pairs, args.judgments, fields, args.rank_method)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
