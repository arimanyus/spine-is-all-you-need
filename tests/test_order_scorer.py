"""Order-bias scorer tests on synthetic fixtures. No models, no downloads, CPU only."""

import json

import pytest

from order import prompt
from order.scorer import (
    UNPARSEABLE, Fields, bradley_terry, checks, flip_score, kendall_tau, load_judgments, load_pairs,
    main, normalize_human, parse_verdict, slot_to_answer, summarize, to_markdown, wilson, win_rates,
)


# ----------------------------------------------------------------------------- prompt

def test_prompt_is_locked():
    assert prompt.SYSTEM.startswith("You are an impartial judge.")
    assert prompt.USER.endswith("Output only one token: A, B, or C (tie).")
    msgs = prompt.build_messages("q", "x", "y")
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "[Question]\nq\n\n[Response A]\nx\n\n[Response B]\ny\n\nOutput only one token: A, B, or C (tie)."
    pair = {"answer1": "x", "answer2": "y"}
    assert prompt.slot_answers(pair, "AB") == ("x", "y")
    assert prompt.slot_answers(pair, "BA") == ("y", "x")


# ----------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("raw, want", [
    ("A", "A"), ("b", "B"), (" C\n", "C"), ("**A**", "A"), ('"B"', "B"), ("A.", "A"),
    ("Answer: B.", "B"), ("Final verdict: C (tie)", "C"), ("C (tie)", "C"), ("Output: A", "A"),
    ("Response A", "A"), ("Response B.", "B"), ("Tie", "C"), ("tie.", "C"), ("(tie)", "C"), ("A tie", "C"),
    ("A\n\nResponse A is better because it is correct.", "A"),
    ("Both responses are good, but Response B is more accurate.", "B"),
    ("I prefer A", "A"), ("The winner is B", "B"), ("B is better.", "B"), ("The answer is C", "C"),
    ("A and B are equally good", "C"), ("Response A and Response B are equal", "C"),
    ("Neither response answers the question.", "C"),
])
def test_parse_verdict(raw, want):
    assert parse_verdict(raw) == want


@pytest.mark.parametrize("raw", ["", "   ", None, "AB", "A or B", "Response A or Response B",
                                 "I cannot decide.", "D", "The first one."])
def test_parse_unparseable(raw):
    assert parse_verdict(raw) == UNPARSEABLE


# ----------------------------------------------------------------------------- mapping

def test_slot_to_answer():
    assert slot_to_answer("A", "AB") == "1" and slot_to_answer("B", "AB") == "2"
    assert slot_to_answer("A", "BA") == "2" and slot_to_answer("B", "BA") == "1"
    assert slot_to_answer("C", "AB") == "tie" and slot_to_answer("C", "BA") == "tie"
    assert slot_to_answer(UNPARSEABLE, "AB") == "unparseable"
    with pytest.raises(ValueError):
        slot_to_answer("A", "XY")


def test_normalize_human():
    assert normalize_human(1) == "1" and normalize_human("2") == "2"
    assert normalize_human("tie") == "tie" and normalize_human(None) is None
    with pytest.raises(ValueError):
        normalize_human("3")


# ----------------------------------------------------------------------------- flips

def test_flip_score():
    assert flip_score("1", "1") == 0.0
    assert flip_score("tie", "tie") == 0.0
    assert flip_score("1", "2") == 1.0
    assert flip_score("1", "tie") == 0.5
    assert flip_score("tie", "2") == 0.5
    assert flip_score("1", "unparseable") is None
    assert flip_score("unparseable", "unparseable") is None


# ----------------------------------------------------------------------------- Wilson

@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0), (1, 100, 0.0018, 0.0545),
])
def test_wilson_known_values(k, n, lo, hi):
    got = wilson(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_empty():
    assert wilson(0, 0) == (0.0, 1.0)


# ----------------------------------------------------------------------------- ranking

def test_kendall_tau_known_values():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(2 / 3)  # one discordant of six
    assert kendall_tau([1, 2, 3], [2, 2, 3]) == pytest.approx(2 / 2.449489742783178)  # tau-b with one tie
    assert kendall_tau([1, 1], [1, 2]) is None


def test_win_rates_and_bt_agree_on_order():
    outcomes = [("x", "y", "1"), ("x", "z", "1"), ("y", "z", "1"), ("y", "z", "tie")]
    wr = win_rates(outcomes)
    assert wr == {"x": 1.0, "y": 0.5, "z": pytest.approx(0.5 / 3)}
    bt = bradley_terry(outcomes)
    assert bt["x"] > bt["y"] > bt["z"] and sum(bt.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- fixtures

MODELS = ["m1", "m2", "m3", "m4"]


def make_pairs():
    """Set M: 18 pairs over 4 models (every model pair x 3), human winner = model1 twice, tie once,
    so the human ranking is m1 > m2 > m3 > m4. Set F: 4 identical controls + 4 near-ties."""
    pairs, k = [], 0
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            for human in ("1", "1", "tie"):
                pairs.append({"id": f"m{k}", "set": "M", "stratum": "clear" if human == "1" else "tie",
                              "question": "q", "answer1": f"ans-{a}", "answer2": f"ans-{b}",
                              "model1": a, "model2": b, "human_winner": human})
                k += 1
    for k in range(4):
        pairs.append({"id": f"f{k}", "set": "F", "stratum": "identical", "question": "q",
                      "answer1": "same", "answer2": "same", "model1": None, "model2": None, "human_winner": None})
    for k in range(4, 8):
        pairs.append({"id": f"f{k}", "set": "F", "stratum": "near_tie", "question": "q",
                      "answer1": "x", "answer2": "y", "model1": None, "model2": None, "human_winner": None})
    return pairs


def judge_rows(name, verdict_fn, logprobs=None):
    """One judgment row per (pair, order) for a synthetic judge. verdict_fn(pair, order) -> raw text."""
    rows = []
    for p in make_pairs():
        for order in ("AB", "BA"):
            row = {"pair_id": p["id"], "judge": name, "order": order, "raw": verdict_fn(p, order)}
            if logprobs:
                row["logprob_a"], row["logprob_b"] = logprobs(p, order)
            rows.append(row)
    return rows


def oracle(p, order):
    """Order-invariant judge: follows human_winner, says tie on set F."""
    w = p["human_winner"] or "tie"
    if w == "tie":
        return "C"
    return "A" if (w == "1") == (order == "AB") else "B"


def always_a(p, order):
    return "A"


def run(rows, pairs=None):
    return summarize(load_pairs(pairs or make_pairs()), load_judgments(rows))


# ----------------------------------------------------------------------------- end to end

def test_oracle_judge_has_no_flips_and_passes_checks():
    rep = run(judge_rows("oracle", oracle))
    j = rep["judges"]["oracle"]
    all_ = j["sets"]["all"]
    assert all_["n_pairs"] == 26 and all_["n_judgments"] == 52
    assert all_["full_flip"]["k"] == 0 and all_["any_flip"]["k"] == 0 and all_["tie_mismatch"] == 0
    assert all_["unparseable"]["k"] == 0
    assert j["sets"]["M"]["accuracy"]["AB"]["rate"] == 1.0 and j["sets"]["M"]["accuracy"]["BA"]["rate"] == 1.0
    assert j["sets"]["M"]["ranking"]["kendall_tau"] == pytest.approx(1.0)
    assert j["sets"]["M"]["ranking"]["ranking"]["AB"] == MODELS
    assert j["sets"]["F"]["ranking"] is None  # no models on set F
    assert set(j["sets"]["M"]["strata"]) == {"clear", "tie"} and set(j["sets"]["F"]["strata"]) == {"identical", "near_tie"}
    # Identical controls judged tie: slot-A preference undefined, so the check is not evaluable.
    assert j["identical_controls"]["slot_a_pref"]["n"] == 0
    c = j["checks"]
    assert c["per_verdict_null"]["pass"] is None
    assert c["ranking_null"] == {"pass": True, "kendall_tau": pytest.approx(1.0)}
    assert c["leakage_signal"]["signal"] is False


def test_always_a_judge_flips_everything_and_fails():
    j = run(judge_rows("slotA", always_a))["judges"]["slotA"]
    all_ = j["sets"]["all"]
    assert all_["full_flip"]["rate"] == 1.0 and all_["slot_a_pref"]["rate"] == 1.0
    assert j["identical_controls"]["slot_a_pref"] == {"k": 8, "n": 8, "rate": 1.0, "ci_lo": pytest.approx(0.676, abs=1e-3), "ci_hi": 1.0}
    assert j["sets"]["M"]["accuracy"]["AB"]["rate"] == pytest.approx(12 / 18)  # answer1 == human winner on 'clear'
    assert j["sets"]["M"]["accuracy"]["BA"]["rate"] == 0.0
    # model1 always wins in AB and always loses in BA -> the ranking inverts.
    rk = j["sets"]["M"]["ranking"]
    assert rk["ranking"]["AB"] == MODELS and rk["ranking"]["BA"] == MODELS[::-1]
    assert rk["kendall_tau"] == pytest.approx(-1.0)
    c = j["checks"]
    assert c["per_verdict_null"]["pass"] is False and c["per_verdict_null"]["slot_a_pref_identical"] == 1.0
    assert c["ranking_null"]["pass"] is False
    assert c["leakage_signal"]["signal"] is False  # F == M == 1.0, not greater


def test_identical_controls_balanced_judge_passes_slot_check():
    def fair(p, order):
        if p["stratum"] == "identical":  # half A, half B across the 8 judgments
            return "A" if (int(p["id"][1:]) + (order == "BA")) % 2 == 0 else "B"
        return oracle(p, order)

    j = run(judge_rows("fair", fair))["judges"]["fair"]
    ident = j["identical_controls"]
    assert ident["slot_a_pref"]["rate"] == 0.5
    # A in AB and B in BA both pick answer1, so the verdict is order-invariant: no flips.
    assert ident["full_flip"]["k"] == 0
    assert j["checks"]["per_verdict_null"]["pass"] is True


def test_half_flips_and_tie_mismatch():
    def hedger(p, order):  # decisive in AB, tie in BA on set M 'clear' pairs
        if p["set"] == "M" and p["stratum"] == "clear" and order == "BA":
            return "C"
        return oracle(p, order)

    j = run(judge_rows("hedger", hedger))["judges"]["hedger"]
    m = j["sets"]["M"]
    assert m["full_flip"]["k"] == 0 and m["any_flip"]["k"] == 12 and m["tie_mismatch"] == 12
    assert m["mean_flip_score"] == pytest.approx(0.5 * 12 / 18)
    assert m["strata"]["clear"]["any_flip"]["rate"] == 1.0 and m["strata"]["tie"]["any_flip"]["rate"] == 0.0
    c = j["checks"]["per_verdict_null"]
    assert c["full_flip_rate"] == 0.0 and c["any_flip_rate"] == pytest.approx(12 / 26)


def test_unparseables_are_counted_not_dropped():
    def flaky(p, order):
        if p["id"] in ("m0", "f0") and order == "BA":
            return "I cannot decide between them."
        return oracle(p, order)

    rows = judge_rows("flaky", flaky)
    j = run(rows)["judges"]["flaky"]
    all_ = j["sets"]["all"]
    assert all_["unparseable"] == {"k": 2, "n": 52, "rate": pytest.approx(2 / 52), "ci_lo": pytest.approx(0.0106, abs=1e-3),
                                   "ci_hi": pytest.approx(0.1298, abs=1e-3)}
    assert all_["pairs_scored_for_flip"] == 24 and all_["full_flip"]["n"] == 24
    # Accuracy keeps the unparseable judgment in the denominator as a miss.
    assert j["sets"]["M"]["accuracy"]["BA"] == {"k": 17, "n": 18, "rate": pytest.approx(17 / 18),
                                                "ci_lo": pytest.approx(0.7424, abs=1e-3), "ci_hi": pytest.approx(0.9901, abs=1e-3)}
    assert j["sets"]["M"]["accuracy"]["AB"]["rate"] == 1.0


def test_missing_judgment_counts_as_unparseable():
    rows = [r for r in judge_rows("partial", oracle) if not (r["pair_id"] == "m1" and r["order"] == "AB")]
    all_ = run(rows)["judges"]["partial"]["sets"]["all"]
    assert all_["missing_judgments"] == 1 and all_["unparseable"]["k"] == 1 and all_["pairs_scored_for_flip"] == 25


def test_duplicate_judgment_rejected():
    rows = judge_rows("dup", oracle)
    with pytest.raises(ValueError):
        load_judgments(rows + rows[:1])


def test_delta_logprob_shift():
    bias = 0.7

    def lp(p, order):
        # Answer-identity margin +1.0 for answer1 on every pair, plus a constant lean towards slot A.
        margin = 1.0 if order == "AB" else -1.0
        d = margin + bias
        return d, 0.0  # logP(A) - logP(B) == d

    def by_lp(p, order):
        return "A" if lp(p, order)[0] > 0 else "B"

    dl = run(judge_rows("lp", by_lp, lp))["judges"]["lp"]["sets"]["all"]["delta_logprob"]
    assert dl["n"] == 26
    assert dl["shift_mean"] == pytest.approx(2 * bias) and dl["shift_median"] == pytest.approx(2 * bias)
    assert dl["slot_margin_mean"]["AB"] == pytest.approx(1.0 + bias)
    assert dl["slot_margin_mean"]["BA"] == pytest.approx(-1.0 + bias)


def test_delta_logprob_absent_is_reported_as_none():
    dl = run(judge_rows("nolp", oracle))["judges"]["nolp"]["sets"]["all"]["delta_logprob"]
    assert dl == {"n": 0, "shift_mean": None, "shift_median": None, "slot_margin_mean": {"AB": None, "BA": None}}


def test_ranking_reshuffle_fails_ranking_null():
    def reversed_in_ba(p, order):  # correct in AB, reversed in BA -> rankings invert
        v = oracle(p, order)
        return v if order == "AB" or v == "C" else {"A": "B", "B": "A"}[v]

    j = run(judge_rows("rev", reversed_in_ba))["judges"]["rev"]
    rk = j["sets"]["M"]["ranking"]
    assert rk["ranking"]["AB"] == MODELS and rk["ranking"]["BA"] == MODELS[::-1]
    assert rk["kendall_tau"] == pytest.approx(-1.0)
    assert j["checks"]["ranking_null"]["pass"] is False


def test_leakage_signal():
    def leaky(p, order):  # perfect on set M, order-driven on set F
        return oracle(p, order) if p["set"] == "M" else "A"

    c = run(judge_rows("leaky", leaky))["judges"]["leaky"]["checks"]["leakage_signal"]
    assert c == {"signal": True, "flip_F": 1.0, "flip_M": 0.0}


def test_checks_not_evaluable_without_sets():
    all_ = run(judge_rows("x", oracle))["judges"]["x"]["sets"]["all"]
    c = checks(all_, None, None, None)
    assert c["per_verdict_null"]["pass"] is None and c["leakage_signal"]["signal"] is None


def test_bradley_terry_ranking_option():
    rep = summarize(load_pairs(make_pairs()), load_judgments(judge_rows("o", oracle)), rank_method="bt")
    rk = rep["judges"]["o"]["sets"]["M"]["ranking"]
    assert rk["method"] == "bt" and rk["ranking"]["AB"] == MODELS and rk["kendall_tau"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- schema + IO

def test_configurable_field_names():
    f = Fields(id="pid", human_winner="gold", pair_id="pid", raw="text", identical_stratum="same_answer")
    pairs = [{"pid": "p1", "set": "F", "stratum": "same_answer", "answer1": "x", "answer2": "x", "gold": None}]
    rows = [{"pid": "p1", "judge": "j", "order": o, "text": "A"} for o in ("AB", "BA")]
    rep = summarize(load_pairs(pairs, f), load_judgments(rows, f))
    j = rep["judges"]["j"]
    assert j["identical_controls"]["slot_a_pref"]["rate"] == 1.0 and j["sets"]["F"]["full_flip"]["rate"] == 1.0


def test_identical_detected_from_equal_answers_without_stratum():
    pairs = load_pairs([{"id": "p", "set": "F", "stratum": "whatever", "answer1": "x", "answer2": "x"}])
    assert pairs[0]["identical"] is True


def test_cli_writes_json_and_markdown(tmp_path):
    pairs_path, judg_path = tmp_path / "pairs.jsonl", tmp_path / "judgments.jsonl"
    pairs_path.write_text("\n".join(json.dumps(p) for p in make_pairs()), encoding="utf-8")
    rows = judge_rows("oracle", oracle) + judge_rows("slotA", always_a)
    judg_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    main(["--pairs", str(pairs_path), "--judgments", str(judg_path), "--out", str(tmp_path / "out")])
    rep = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert set(rep["judges"]) == {"oracle", "slotA"} and rep["fields"]["human_winner"] == "human_winner"
    md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert md == to_markdown(rep)
    assert "| slotA | F | identical |" in md and "## Pre-registered checks" in md
    assert "| slotA | FAIL" in md and "| oracle | n/a" in md
