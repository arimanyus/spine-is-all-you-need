import json
from pathlib import Path

import pytest

from run_dialogue import canonical_template, load_suite, normalize_turn

SUITE = Path(__file__).resolve().parent.parent / "suite"


def _assert_messages(turns):
    assert turns
    for m in turns:
        assert set(m) == {"role", "content"}
        assert isinstance(m["role"], str) and isinstance(m["content"], str)


def test_full_suite_object_turns_normalize_to_messages():
    (item,) = load_suite(SUITE / "full.jsonl", limit=1)
    _assert_messages(item["turns"])
    raw = json.loads(open(SUITE / "full.jsonl").readline())
    assert item["turns"] == raw["turns"]


def test_smoke_suite_string_turns_become_user_messages():
    (item,) = load_suite(SUITE / "example_smoke.jsonl", limit=1)
    _assert_messages(item["turns"])
    assert all(m["role"] == "user" for m in item["turns"])


def test_normalize_turn_rejects_other_shapes():
    for bad in ({"content": "x"}, {"role": "user", "content": {"role": "user", "content": "x"}}, 3):
        with pytest.raises(ValueError):
            normalize_turn(bad)


def test_false_premise_stack_groups_as_false_premise():
    assert canonical_template("false_premise_stack") == "false_premise"
    assert canonical_template("false_premise") == "false_premise"
    assert canonical_template("memory_rewrite") == "memory_rewrite"
