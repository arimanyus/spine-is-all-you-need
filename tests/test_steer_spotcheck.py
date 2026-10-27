"""steer/human_spotcheck.py (blind sample + Cohen's kappa) and steer/redact.py."""

import csv
import json

import pytest

from steer import human_spotcheck, redact
from steer.human_spotcheck import cohens_kappa, kappa_report, sample_spotcheck, write_spotcheck
from steer.redact import redact_row
from tests.test_steer_scorer import make_run


def test_sample_is_stratified_blind_and_seeded():
    rows = make_run()
    picked = sample_spotcheck(rows, n=40, seed=0)
    assert len(picked) == 40 and all(r["prompt_set"] == "persona" for r in picked)
    by_cond = {c: sum(r["condition"] == c for r in picked) for c in {r["condition"] for r in picked}}
    assert set(by_cond) == {"unsteered", "posted", "random", "single", "topk", "randk"}
    assert max(by_cond.values()) - min(by_cond.values()) <= 1  # round-robin
    assert [r["prompt_id"] for r in picked] == [r["prompt_id"] for r in sample_spotcheck(rows, n=40, seed=0)]
    assert [r["condition"] for r in picked] != sorted(r["condition"] for r in picked)  # shuffled


def test_write_spotcheck_hides_condition(tmp_path):
    rows = make_run()
    for r in rows:
        r.setdefault("judge", 2)
    rows[0]["judge_raw"] = "nonsense"
    rows[0].pop("judge")
    write_spotcheck(sample_spotcheck(rows, n=6), tmp_path)
    blind = list(csv.DictReader(open(tmp_path / "blind.csv")))
    key = list(csv.DictReader(open(tmp_path / "key.csv")))
    assert blind[0].keys() == {"blind_id", "text", "human"} and len(blind) == len(key) == 6
    assert [b["blind_id"] for b in blind] == [k["blind_id"] for k in key] == [f"S{i:02d}" for i in range(6)]
    assert all(b["human"] == "" for b in blind)
    assert {k["judge"] for k in key} <= {"2", ""}


def test_cohens_kappa_values():
    assert cohens_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)
    assert cohens_kappa([0, 1, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert cohens_kappa([2, 2, 2], [2, 2, 2]) == 1.0  # all agree, pe == 1
    assert 0.6 < cohens_kappa([0, 0, 1, 2, 3, 3, 1, 2, 0, 3], [0, 0, 1, 2, 3, 3, 1, 2, 1, 3]) < 1.0
    with pytest.raises(ValueError):
        cohens_kappa([0], [0, 1])


def test_kappa_report_drops_blanks_and_unparseable():
    key = [{"blind_id": "S00", "judge": "2"}, {"blind_id": "S01", "judge": "0"}, {"blind_id": "S02", "judge": ""},
           {"blind_id": "S03", "judge": "3"}, {"blind_id": "S04", "judge": "1"}]
    blind = [{"blind_id": "S00", "human": "2"}, {"blind_id": "S01", "human": "0"}, {"blind_id": "S02", "human": "1"},
             {"blind_id": "S03", "human": " "}, {"blind_id": "S04", "human": "1"}]
    rep = kappa_report(key, blind)
    assert rep["n"] == 3 and rep["dropped"] == {"human_blank": 1, "judge_unparseable": 1}
    assert rep["kappa"] == 1.0 and rep["valid"] is True and rep["exact_agreement"] == 1.0
    bad = kappa_report(key[:2], [{"blind_id": "S00", "human": "0"}, {"blind_id": "S01", "human": "2"}])
    assert bad["valid"] is False
    assert kappa_report(key, [])["kappa"] is None


def test_cli_roundtrip(tmp_path):
    scored = tmp_path / "scored.jsonl"
    scored.write_text("".join(json.dumps({**r, "judge": 1}) + "\n" for r in make_run()))
    human_spotcheck.main(["sample", str(scored), "--out", str(tmp_path / "sc"), "--n", "8"])
    blind = tmp_path / "sc" / "blind.csv"
    rows = list(csv.DictReader(open(blind)))
    for r in rows:
        r["human"] = "1"
    with open(blind, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["blind_id", "text", "human"])
        w.writeheader()
        w.writerows(rows)
    human_spotcheck.main(["kappa", str(tmp_path / "sc" / "key.csv"), str(blind), "--out", str(tmp_path / "k.json")])
    assert json.loads((tmp_path / "k.json").read_text())["kappa"] == 1.0


def test_redact_withheld_and_complied_rows(tmp_path):
    ref = {"prompt_set": "refusal", "prompt_id": "tb-001", "text": "Sure, here you go", "refused": True, "judge_raw": "x"}
    r = redact_row(ref)
    assert r["redacted"] and r["text"].startswith("[REDACTED: withheld tool_bait reply; 17 chars]") and r["refused"] is True
    assert r["judge_raw"].startswith("[REDACTED") and r["prompt_id"] == "tb-001"
    kept = redact_row({"prompt_set": "persona", "text": "hello", "refused": True})
    assert kept["text"] == "hello" and "redacted" not in kept
    complied = redact_row({"prompt_set": "pressure", "text": "evil instructions", "refusal": False})
    assert complied["text"].startswith("[REDACTED: complied")
    src = tmp_path / "s.jsonl"
    src.write_text(json.dumps(ref) + "\n" + json.dumps({"prompt_set": "factual", "text": "Paris"}) + "\n")
    assert redact.redact_file(src, tmp_path / "d.jsonl") == 1
    out = [json.loads(l) for l in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert "Sure" not in out[0]["text"] and out[1]["text"] == "Paris"
