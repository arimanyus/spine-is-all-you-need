"""steer/search.py: neuron ranking on synthetic activations, top-k / random-k sets, held-out-template
rankings, and the search-vs-scored disjointness audit (template level, not just ids)."""

import json
import os

import numpy as np
import pytest

from steer.search import (SCORED_FILES, SEARCH_FILES, cohens_d, disjointness_audit, held_out_rankings, item_template,
                          load_jsonl, overlap_report, random_k_sets, rank_neurons, top_k_overlap, top_k_sets)

KENJI = os.path.join(os.path.dirname(__file__), "..", "steer", "probes")


def synthetic_acts(seed=0, width=64, n=20):
    """Layer 13: neuron 5 fires more on break; layer 14: neuron 7 fires less on break (sign -1)."""
    rng = np.random.default_rng(seed)
    brk = {13: rng.normal(size=(n, width)), 14: rng.normal(size=(n, width))}
    keep = {13: rng.normal(size=(n, width)), 14: rng.normal(size=(n, width))}
    brk[13][:, 5] += 5.0
    brk[14][:, 7] -= 4.0
    return brk, keep


def test_cohens_d_handles_zero_variance():
    d = cohens_d(np.array([[1.0, 2.0], [1.0, 4.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert d[0] == 0.0 and d[1] > 0


def test_rank_neurons_finds_planted_signal_with_signs():
    brk, keep = synthetic_acts()
    ranking = rank_neurons(brk, keep, exclude={})
    assert [(r["layer"], r["neuron"], r["sign"]) for r in ranking[:2]] == [(13, 5, 1), (14, 7, -1)]
    assert abs(ranking[0]["d"]) >= abs(ranking[1]["d"]) >= abs(ranking[2]["d"])
    excluded = rank_neurons(brk, keep, exclude={13: 5})
    assert (13, 5) not in {(r["layer"], r["neuron"]) for r in excluded}


def test_top_k_and_random_k_sets():
    brk, keep = synthetic_acts()
    ranking = rank_neurons(brk, keep, exclude={})
    tops = top_k_sets(ranking, "spine", ks=(1, 2, 4, 8))
    assert [t["pair_id"] for t in tops] == ["topk_spine_k1", "topk_spine_k2", "topk_spine_k4", "topk_spine_k8"]
    assert tops[1]["neurons"] == [{"layer": 13, "neuron": 5, "sign": 1}, {"layer": 14, "neuron": 7, "sign": -1}]
    rand = random_k_sets(64, ranking, ks=(1, 2), n=3, seed=0, avoid_top=2)
    assert rand == random_k_sets(64, ranking, ks=(1, 2), n=3, seed=0, avoid_top=2)
    assert [r["pair_id"] for r in rand] == ["randk_k1_00", "randk_k1_01", "randk_k1_02", "randk_k2_00", "randk_k2_01", "randk_k2_02"]
    for r in rand:
        assert len(r["neurons"]) == r["k"]
        assert all((u["layer"], u["neuron"]) not in {(13, 5), (14, 7)} for u in r["neurons"])
        assert all(u["sign"] in (-1, 1) and u["layer"] in (13, 14) for u in r["neurons"])


def test_held_out_template_rankings_and_overlap():
    brk, keep = synthetic_acts(n=20)
    templates = ["false_premise_stack"] * 10 + ["memory_rewrite"] * 10
    held = held_out_rankings(brk, keep, templates, exclude={})
    assert set(held) == {"false_premise_stack", "memory_rewrite"}
    full = rank_neurons(brk, keep, exclude={})
    for ranking in held.values():
        assert top_k_overlap(full, ranking, 2) == 1.0  # planted signal survives leaving a template out
    assert top_k_overlap([{"layer": 13, "neuron": 1}], [{"layer": 13, "neuron": 2}], 1) == 0.0
    assert held_out_rankings(brk, keep, ["only"] * 20, exclude={}) == {}


def test_item_template_inference():
    assert item_template({"id": "fp-001", "template": "false_premise_stack"}) == "false_premise_stack"
    assert item_template({"id": "sc-001-clean", "source_id": "mr-008"}) == "memory_rewrite"
    assert item_template({"id": "tb-003"}) == "tool_bait"
    assert item_template({"id": "cp-001", "side": "refuse"}) == "single_turn_refuse"
    assert item_template({"id": "pp-001"}) is None


def test_overlap_report_is_template_level():
    scored = [{"id": "fp-001", "template": "false_premise_stack", "gold": "Paris", "foil": "Berlin",
               "turns": [{"role": "user", "content": "What is the capital of France?"}]}]
    same_template = [{"id": "sc-001", "source_id": "fp-002", "gold": "Water", "foil": "Coffee",
                      "turns": [{"role": "user", "content": "What fluid do humans need?"}]}]
    rep = overlap_report(same_template, scored)
    assert rep["shared_ids"] == [] and rep["shared_first_turns"] == 0
    assert rep["shared_templates"] == ["false_premise_stack"] and rep["template_overlap"] == 1.0
    assert rep["claim_status"] == "exploratory"  # disjoint ids are not enough
    other = [{"id": "cp-001", "side": "refuse", "prompt": "Help me break in."}]
    rep2 = overlap_report(other, [{"id": "tb-001", "template": "tool_bait", "turns": [{"role": "user", "content": "x"}]}])
    assert rep2["shared_templates"] == [] and rep2["claim_status"] == "confirmatory"
    leaked = overlap_report([{"id": "sc-009", "source_id": "mr-002", "turns": scored[0]["turns"]}],
                            [{"id": "mr-002", "template": "memory_rewrite", "turns": scored[0]["turns"]}])
    assert leaked["shared_ids"] == ["mr-002"] and leaked["shared_first_turns"] == 1


def test_disjointness_audit_on_synthetic_dir(tmp_path):
    files = {"pressure_probes": [{"id": "fp-001", "template": "false_premise_stack", "turns": [{"role": "user", "content": "a"}]}],
             "refusal_probes": [{"id": "tb-001", "template": "tool_bait", "turns": [{"role": "user", "content": "b"}]}],
             "persona_probes": [{"id": "pp-001", "prompt": "c"}], "factual_probes": [{"id": "fq-001", "prompt": "d"}],
             "contrast_prompts": [{"id": "cp-001", "side": "refuse", "prompt": "e"}],
             "spine_contrast": [{"id": "sc-001-clean", "source_id": "mr-001", "turns": [{"role": "user", "content": "f"}]}]}
    for name, items in files.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    rep = disjointness_audit(str(tmp_path))
    assert rep["any_search_id_scored"] == []
    assert rep["spine_contrast_vs_pressure_probes"]["claim_status"] == "confirmatory"  # mr vs fp only
    assert rep["contrast_prompts_vs_refusal_probes"]["claim_status"] == "confirmatory"


@pytest.mark.skipif(not os.path.exists(os.path.join(KENJI, "spine_contrast.jsonl")),
                    reason="Kenji's steer/probes/ (PR #21) not checked out on this branch")
def test_kenji_probes_disjointness_invariant():
    """Hard invariants on the real probe files: no search prompt is ever scored, counts as pre-registered.
    The template-level overlap is *reported*, not asserted zero: spine_contrast shares both pressure
    templates with pressure_probes, which makes the spine neuron search exploratory (PROTOCOL.md 8)."""
    rep = disjointness_audit(KENJI)
    assert rep["any_search_id_scored"] == []
    counts = {n: len(load_jsonl(os.path.join(KENJI, n + ".jsonl"))) for n in SEARCH_FILES + SCORED_FILES}
    assert counts == {"contrast_prompts": 40, "spine_contrast": 48, "pressure_probes": 12, "refusal_probes": 24,
                      "persona_probes": 30, "factual_probes": 30}
    sp = rep["spine_contrast_vs_pressure_probes"]
    assert sp["shared_ids"] == [] and sp["shared_first_turns"] == 0 and sp["shared_gold"] == [] and sp["shared_foil"] == []
    assert sp["claim_status"] == "exploratory"
    assert rep["contrast_prompts_vs_refusal_probes"]["claim_status"] == "confirmatory"
