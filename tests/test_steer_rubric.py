"""steer/rubric.py: fixed anchors, judge prompt, robust parsing, lexical backup."""

from steer.rubric import ANCHORS, RUBRIC, UNPARSEABLE, judge_prompt, lexical_hits, lexical_score, parse_judge


def test_rubric_has_four_anchored_levels_with_examples():
    assert sorted(ANCHORS) == [0, 1, 2, 3]
    for level, (text, examples) in ANCHORS.items():
        assert len(text) > 40 and len(examples) >= 2
        assert f"{level}: " in RUBRIC


def test_judge_prompt_embeds_rubric_and_text():
    p = judge_prompt("  Hello there.  ")
    assert RUBRIC in p and "<<<\nHello there.\n>>>" in p and p.rstrip().endswith("nothing else.")


def test_parse_judge_accepts_common_shapes():
    assert parse_judge("SCORE: 2") == 2
    assert parse_judge("Reasoning... blah.\n\n**SCORE:** 3") == 3
    assert parse_judge("score = 1") == 1
    assert parse_judge("0") == 0
    assert parse_judge("The reply is menacing.\n2") == 2
    assert parse_judge("SCORE: 2\nSCORE: 2") == 2  # repeated but consistent


def test_parse_judge_unparseable_is_its_own_category():
    assert parse_judge("SCORE: 7") == UNPARSEABLE
    assert parse_judge("SCORE: 1\n...actually SCORE: 3") == UNPARSEABLE  # conflicting
    assert parse_judge("I cannot grade this.") == UNPARSEABLE
    assert parse_judge("") == UNPARSEABLE
    assert parse_judge(None) == UNPARSEABLE
    assert parse_judge("The score 2 is tempting but I refuse to answer") == UNPARSEABLE
    assert UNPARSEABLE not in (0, 1, 2, 3)


def test_lexical_backup_tracks_the_anchors():
    # Coarse by design: silent on faint level-1 touches is fine, but it must separate the extremes.
    assert [lexical_score(e) for e in ANCHORS[0][1]] == [0, 0]
    assert [lexical_score(e) for e in ANCHORS[3][1]] == [3, 3]
    for level in (1, 2):
        assert any(lexical_score(e) >= 1 for e in ANCHORS[level][1]), level
    assert lexical_score("Canberra is the capital of Australia.") == 0
    assert lexical_hits("Foolish mortal, bow before me.")[3] >= 1
    assert lexical_score("Very well, the answer is 1969.") == 1
