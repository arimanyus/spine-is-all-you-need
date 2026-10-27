"""steer/controls.py: seeded random pairs and the normalized dose math, synthetic arrays only."""

import json

import numpy as np
import pytest

from steer.controls import (POSTED, POSTED_PAIR, dose_scale, main, median_resid_norm, sample_random_pairs,
                            single_neuron_ablations, steering_vector, write_pairs)


def test_posted_pair_is_the_claim():
    assert POSTED_PAIR["neurons"] == [{"layer": 13, "neuron": 7956, "sign": -1}, {"layer": 14, "neuron": 8949, "sign": 1}]


def test_random_pairs_are_deterministic_and_well_formed():
    a = sample_random_pairs(9728, n=10, seed=0)
    assert a == sample_random_pairs(9728, n=10, seed=0)
    assert a != sample_random_pairs(9728, n=10, seed=1)
    assert [p["pair_id"] for p in a] == [f"rand_{i:02d}" for i in range(10)]
    for p in a:
        assert [(u["layer"], u["sign"]) for u in p["neurons"]] == [(13, -1), (14, 1)]
        for u in p["neurons"]:
            assert 0 <= u["neuron"] < 9728 and u["neuron"] != POSTED[u["layer"]]
    for layer in (13, 14):
        picks = [u["neuron"] for p in a for u in p["neurons"] if u["layer"] == layer]
        assert len(set(picks)) == 10  # no neuron reused within a layer


def test_random_pairs_exclude_posted_even_in_a_tiny_mlp():
    pairs = sample_random_pairs(POSTED[14] + 1, n=POSTED[14] - 2, seed=3)
    assert all(u["neuron"] != POSTED[u["layer"]] for p in pairs for u in p["neurons"])
    with pytest.raises(ValueError):
        sample_random_pairs(5, n=6)


def test_single_neuron_ablations():
    assert [s["pair_id"] for s in single_neuron_ablations()] == ["single_L13N7956", "single_L14N8949"]
    assert all(len(s["neurons"]) == 1 for s in single_neuron_ablations())


def test_median_skips_first_position():
    norms = np.array([1000.0, 10.0, 12.0, 14.0])
    assert median_resid_norm(norms) == 12.0
    assert median_resid_norm(np.array([[1000.0, 10.0, 20.0], [999.0, 30.0, 40.0]])) == 25.0
    assert median_resid_norm(norms, skip=0) == 13.0
    with pytest.raises(ValueError):
        median_resid_norm(np.array([5.0]))


def test_dose_scale_and_vector_norm():
    rng = np.random.default_rng(0)
    col = rng.normal(size=2560)
    norms = np.concatenate([[500.0], rng.uniform(40, 60, size=200)])
    med = float(np.median(norms[1:]))
    assert dose_scale(1.0, norms, col) == pytest.approx(med / np.linalg.norm(col))
    assert dose_scale(0.5, norms, np.linalg.norm(col)) == pytest.approx(0.5 * med / np.linalg.norm(col))
    for dose in (0.25, 0.5, 1.0, 2.0):
        assert np.linalg.norm(steering_vector(dose, norms, col)) == pytest.approx(dose * med)
    v = steering_vector(1.0, norms, col)
    assert np.allclose(v / np.linalg.norm(v), col / np.linalg.norm(col))  # along the column
    with pytest.raises(ValueError):
        dose_scale(1.0, norms, np.zeros(4))


def test_torch_like_inputs_are_accepted():
    class Fake:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.a, dtype=dtype)

    assert median_resid_norm(Fake([9.0, 1.0, 3.0])) == 2.0
    assert dose_scale(1.0, Fake([9.0, 1.0, 3.0]), Fake([0.0, 2.0])) == 1.0


def test_write_pairs_json(tmp_path):
    path = tmp_path / "pairs.json"
    out = write_pairs(path, 9728, n=3, seed=7)
    saved = json.loads(path.read_text())
    assert saved == out and saved["seed"] == 7 and len(saved["random"]) == 3
    assert saved["posted"] == POSTED_PAIR and len(saved["single"]) == 2
    main(["--intermediate-size", "9728", "--n", "2", "--out", str(tmp_path / "cli.json")])
    assert len(json.loads((tmp_path / "cli.json").read_text())["random"]) == 2
