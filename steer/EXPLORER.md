# neuron-explorer (MIT) cross-check

## Hook / dose equivalence

Hook and dose units vs `https://github.com/lyramakesmusic/neuron-explorer (MIT)` (sweep/check_equivalence.py (HF path only; vLLM and capture checks skipped)):

| layer:neuron | greedy agree input-hook vs output-column | fp32 rel err (input vs output formula) | added MLP-out norm | dose-1 target (resid_p50) |
|---|---|---|---|---|
| L10:n3551 | 16/16 | 4.1e-07 | 9.30 | 9.30 |
| L13:n7956 | 15/16 | 2.0e-07 | 9.38 | 9.38 |
| L14:n8949 | 16/16 | 1.2e-07 | 10.53 | 10.53 |

Dose units: our `steer.py` dose equals explorer's `--dose-unit rel` dose and their recorded `rel_push` — `add_to_activation = dose * resid_p50[L] / ||W_down[L][:, n]||`, so the added MLP-output vector has norm `dose * resid_p50[L]` (a fixed fraction `dose` of the median residual norm leaving layer L). Their absmax dose for the same push is `dose * resid_p50[L] / (||W_down[L][:, n]|| * abs_max[L, n])` (abs_max from their `act_scale.npz`). steer_index / connections were built at rel dose 0.8. Per-neuron conversions:

| layer:neuron | our dose (rel) | add to activation | explorer rel_push | explorer dose (absmax) |
|---|---|---|---|---|
| L10:n3551 | 0.5 | 10.74 | 0.50 | 9.68 |
| L10:n3551 | 0.8 | 17.19 | 0.80 | 15.49 |
| L10:n3551 | 1.0 | 21.48 | 1.00 | 19.36 |
| L13:n7956 | 0.5 | 11.21 | 0.50 | 9.57 |
| L13:n7956 | 0.8 | 17.94 | 0.80 | 15.31 |
| L13:n7956 | 1.0 | 22.42 | 1.00 | 19.13 |
| L14:n8949 | 0.5 | 11.33 | 0.50 | 12.95 |
| L14:n8949 | 0.8 | 18.13 | 0.80 | 20.72 |
| L14:n8949 | 1.0 | 22.66 | 1.00 | 25.90 |

