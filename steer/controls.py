"""Controls for the two-neuron steering claim: random neuron pairs and the normalized dose scale.

The posted recipe adds, at the down_proj output of layer L and at every position,

    dose * median_t>=1(||resid_L[t]||) / ||W_down[:, n]||  *  W_down[:, n]

so the added vector has norm `dose * median residual norm` whatever the neuron's column norm is.
`dose_scale` is that scalar coefficient; `steering_vector` is the vector. Random control pairs take
one neuron from layer 13 and one from layer 14 with the posted signs (13: -1, 14: +1) so that the
only thing that differs from the posted pair is *which* neurons, at the same normalized dose.

    python -m steer.controls --intermediate-size 9728 --out steer/random_pairs.json
"""

import argparse
import json
import random

import numpy as np

POSTED = {14: 8949, 13: 7956}
SIGNS = {14: +1, 13: -1}
POSTED_PAIR = {"pair_id": "posted",
               "neurons": [{"layer": layer, "neuron": n, "sign": SIGNS[layer]} for layer, n in sorted(POSTED.items())]}


def sample_random_pairs(intermediate_size, n=10, seed=0, exclude=POSTED, signs=SIGNS):
    """`n` pairs, one neuron per layer in `signs`, sampled without replacement per layer with
    `random.Random(seed)`, never a neuron in `exclude`. Deterministic for a given
    (intermediate_size, n, seed)."""
    rng = random.Random(seed)
    layers = sorted(signs)
    columns = {}
    for layer in layers:
        pool = [i for i in range(intermediate_size) if i != exclude.get(layer)]
        if n > len(pool):
            raise ValueError(f"layer {layer}: {n} pairs need {n} distinct neurons, only {len(pool)} available")
        columns[layer] = rng.sample(pool, n)
    return [{"pair_id": f"rand_{i:02d}",
             "neurons": [{"layer": layer, "neuron": columns[layer][i], "sign": signs[layer]} for layer in layers]}
            for i in range(n)]


def single_neuron_ablations(pair=POSTED_PAIR):
    """One condition per neuron of `pair`, the other neuron left alone."""
    return [{"pair_id": f"single_L{u['layer']}N{u['neuron']}", "neurons": [u]} for u in pair["neurons"]]


def write_pairs(path, intermediate_size, n=10, seed=0):
    out = {"intermediate_size": intermediate_size, "seed": seed, "signs": {str(k): v for k, v in SIGNS.items()},
           "posted": POSTED_PAIR, "random": sample_random_pairs(intermediate_size, n, seed),
           "single": single_neuron_ablations()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def _as_array(x):
    # torch tensors (and anything else with .detach/.cpu) without importing torch.
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def median_resid_norm(resid_norms, skip=1):
    """Median of per-position residual norms, dropping the first `skip` positions (BOS / first token,
    whose norm is an outlier). `resid_norms` is 1-D (positions) or 2-D (sequences x positions)."""
    norms = np.atleast_2d(_as_array(resid_norms))[:, skip:]
    if norms.size == 0:
        raise ValueError("need at least one position after the skipped prefix")
    return float(np.median(norms))


def dose_scale(dose, resid_norms, w_down_col, skip=1):
    """Coefficient on W_down[:, n]: dose * median(||resid||) / ||W_down[:, n]||.
    `w_down_col` may be the column (vector) or its norm (scalar)."""
    col = _as_array(w_down_col)
    col_norm = float(np.linalg.norm(col)) if col.ndim else float(col)
    if col_norm <= 0:
        raise ValueError("W_down column norm must be positive")
    return dose * median_resid_norm(resid_norms, skip) / col_norm


def steering_vector(dose, resid_norms, w_down_col, skip=1):
    """The vector added at the down_proj output; its norm is dose * median(||resid||)."""
    col = _as_array(w_down_col)
    return dose_scale(dose, resid_norms, col, skip) * col


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--intermediate-size", type=int, required=True, help="MLP width (W_down has this many columns)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="steer/random_pairs.json")
    a = p.parse_args(argv)
    out = write_pairs(a.out, a.intermediate_size, a.n, a.seed)
    for pair in out["random"]:
        print(pair["pair_id"], [(u["layer"], u["neuron"], u["sign"]) for u in pair["neurons"]])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
