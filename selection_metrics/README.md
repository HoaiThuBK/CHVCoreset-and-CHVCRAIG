# Selection-quality metrics — eps_hat, coverage loss, covering radius

Standalone package measuring **three selection-quality quantities** of the
CHV-CRAIG / CHVCoreset pipeline at the **fresh-init (epoch 0)** model state:

| Symbol | Definition | Meaning |
|---|---|---|
| **eps_hat_c** | `max_{i in I_c} min_{h in H_c} d(g_i, g_h)` | certificate radius of the CHVS4 pool `H_c` |
| **L_c(S_c)** | `sum_{i in I_c} min_{j in S_c} d(g_i, g_j)` | coverage loss — the objective greedy CRAIG minimises |
| **R_c(S_c)** | `max_{i in I_c} min_{j in S_c} d(g_i, g_j)` | covering radius (Gonzalez a-priori net) of the coreset |

The gradient embedding is the logit gradient `z_i = softmax(logits) - onehot(y_i)`.
Each quantity is reported under **four distance conventions** so the
"d vs d^2" and "raw z vs normalised g" choices can be settled from data:

- `raw_d`  = ||z_i - z_j||    (raw Euclidean, what CHVCoreset uses on `Z_c`)
- `raw_d2` = ||z_i - z_j||^2  (raw squared, what the code minimises)
- `cos_d`  = sqrt(2 - 2 g_i^T g_j)  (normalised cosine, manuscript formula)
- `cos_d2` = 2 - 2 g_i^T g_j

## Usage

```bash
pip install -r requirements.txt

# Quick check (NumPy only, ~1 s) — verifies the theoretical invariants:
python selection_metrics.py --selftest

# Full measurement:
python selection_metrics.py \
    --datasets fashionmnist cifar10 covtype \
    --fractions 0.01 0.03 0.05 0.1 \
    --seeds 42 43 44 \
    --alpha 5 \
    --selection-distance d2 \
    --outdir ./selection_metrics_out
```

Add `--device cuda` on a GPU machine (it only speeds up the CIFAR gradient
extraction; the measurement itself is NumPy on CPU).

Key parameters:

- `--alpha`: pool multiplier `k_H = alpha * r_c` (default 5, matching the paper).
- `--selection-distance`: `d2` matches the code, `d` matches the lemmas. The
  CSVs always report **all four** conventions; this flag only changes the
  greedy objective used to select `S_c`.

## Outputs (`--outdir`)

- `selection_metrics_perclass_<dataset>.csv` — one row per (class, fraction, seed):
  `n_c, k_H, r_c, S_subset_H, z_norm_*`, and per convention `<conv>__`:
  `eps_hat_c, covering_radius_Sc, coverage_loss_Sc, coverage_loss_Hc, mean_dist_*`.
- `selection_metrics_summary_<dataset>.csv` — aggregated per (fraction, seed):
  eps_hat (max and weighted mean), covering radius (max), coverage loss (sum).

## Self-test

`--selftest` (pure NumPy, no data) verifies the theoretical invariants:
`S_c ⊆ H_c ⊆ I_c`; `eps_hat_c = R(H_c) <= R(S_c)`; `L(S_c) >= L(H_c)`;
`L(S_c) <= n_c * R(S_c)` — all pass for greedy-on-d and greedy-on-d^2.

## Requirements

See `requirements.txt`. `--selftest` needs only `numpy`; dataset measurement
additionally needs `torch`/`torchvision` (FashionMNIST/CIFAR) and
`scikit-learn` (Covtype). Data is downloaded automatically to `./data`.
