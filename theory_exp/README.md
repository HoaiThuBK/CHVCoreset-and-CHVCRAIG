# Theory-verification experiments (Section 3.1) and alpha/Delta ablation (Section 3.2)

These scripts measure the theory-linked quantities of the CHVCoreset /
CHV-CRAIG paper **using the real experiment pipeline** (`chvcoreset.py`,
`craig.py`, `utils.py` in `../src`), so the numbers are faithful to the
method actually used in the paper.

## Setup

Environment: `python>=3.9`, `torch`, `torchvision`, `numpy`, `scikit-learn`
(same as the training scripts). A GPU is recommended for CIFAR-10 / SVHN.

The scripts add `../chv_craig` to `sys.path` automatically (or pass
`--code_dir` pointing to the pipeline folder).

---

## Section 3.1 — certificate, coverage loss, gradient-matching error

`measure_theory.py` warms up the model (matching the paper: 10 epochs for
FashionMNIST/SVHN, 20 for CIFAR-10), then for each class/fraction/seed runs the
**real Budgeted CHVS4 pool** `H_c` (budget `alpha*r_c`) and the greedy CRAIG coreset
`S_c`, and computes, with the manuscript distance `d(z_i,z_j)=||z_i-z_j||_2`:

- `eps_hat_c` — the a posteriori certificate (Prop. certificate),
- coverage loss `L_c(S_c)` for CHV-CRAIG **and** for plain CRAIG,
- relative gradient-matching error `||sum z - sum gamma z|| / ||sum z||`
  with the count weights, for CHV-CRAIG vs CRAIG,
- selection times `T_pool` (Budgeted CHVS4) and `T_greedy`.

```
python measure_theory.py --datasets fashionmnist cifar10 svhn \
    --fractions 0.01 0.03 0.05 --seeds 42 43 44 --alpha 5 \
    --warmup fashionmnist:10 cifar10:20 svhn:10 \
    --device cuda --outdir out_theory
```
Outputs `out_theory/theory_perclass_<ds>.csv` and `theory_summary_<ds>.csv`.

Candidate-pool comparison (M1 ablation): add `--pool random` for a random pool
of the same size `k_H`, or `--pool ff` for farthest-first; output files get the
suffix `_random` / `_ff` and are never overwritten by the default `chvs4` runs.

Notes:
- By default gradients are computed on a **non-augmented** copy of the training
  set (deterministic). Add `--select-on-augmented` to reproduce the training
  scripts exactly (they select on the augmented `trainset_full`).
- Selection itself (`H_c`, `S_c`) is produced by the real code, which minimises
  the **squared** distance; the reported `eps_hat`/coverage use `d`, which gives
  the *same* Voronoi assignment and weights.

---

## Section 3.2 — Ablation

**(a) alpha, selection quality only (cheap, no training):** `ablation.py`
sweeps `alpha` and reports `eps_hat`, gradient error, coverage-loss ratio,
pool size, and pool time:
```
python ablation.py --dataset fashionmnist --fraction 0.03 \
    --alphas 3 5 10 --seeds 42 43 44 --warmup 10 --device cuda \
    --outdir out_ablation
```
Output: `out_ablation/ablation_alpha_<ds>.csv`.

**(b) alpha and Delta vs accuracy/runtime (needs training):**
`run_train_grid.py` drives the training scripts over the grid and collects
final accuracy + cumulative `T_sel`/`T_tot`:
```
python run_train_grid.py --dataset fashionmnist \
    --alphas 3 5 10 --deltas 40 --fractions 0.03 --seeds 42 43 44 \
    --epochs 100 --warmup 10 \
    --outcsv out_ablation/train_grid_fashionmnist.csv
```
(Use `--deltas 20 40 60` at fixed `--alphas 5` for the Delta ablation.)
Output: `out_ablation/train_grid_<ds>.csv` (+ archived per-run CSVs).

---

## Turn CSVs into LaTeX (paste-ready)

```
python make_latex_tables.py --theory out_theory/theory_summary_*.csv
python make_latex_tables.py --alpha  out_ablation/ablation_alpha_fashionmnist.csv
python make_latex_tables.py --grid   out_ablation/train_grid_fashionmnist.csv
```

`finalize_theory.py` produces, in one command, the exact manuscript table
(`tab:theory_verify`), the two-panel figure `fig_theory_verify.pdf` (with
error bars), and the prose numbers (Spearman correlations, coverage-ratio
ranges) once 3-seed CSVs exist for all datasets:
```
python finalize_theory.py --indir out_theory
```

## What to expect (a sanity guide, not the actual numbers)

- `eps_hat_c` small and stable across seeds; decreasing in `alpha`.
- `grad.err (CHV)` close to `grad.err (CRAIG)` — CHV-CRAIG preserves the
  gradient approximation while searching only `H_c`.
- `L_c^CHV / L_c^CRAIG` close to 1 (slightly >= 1), shrinking as `alpha` grows.

---

## Running on limited hardware (no Colab / no GPU)

Section 3.1 is much lighter than full training — it does **not** train to
convergence, only a short warm-up followed by measurement:

- **FashionMNIST (logistic regression): runs fine on CPU**, a few minutes,
  ~30 MB of data:
  ```
  pip install torch torchvision numpy scikit-learn      # CPU build is enough
  python measure_theory.py --datasets fashionmnist \
      --fractions 0.01 0.03 0.05 --seeds 42 43 44 --alpha 5 \
      --warmup fashionmnist:10 --device cpu \
      --data_root ./data --outdir out_theory
  python make_latex_tables.py --theory out_theory/theory_summary_fashionmnist.csv
  ```
- **CIFAR-10 / SVHN (ResNet-20):** the only heavy part is the ResNet warm-up.
  Valid ways to reduce cost (`eps_hat_c`/coverage/grad-error can be measured at
  any checkpoint): reduce the warm-up (`--warmup cifar10:3 svhn:3`), or use
  fewer seeds/fractions (`--seeds 42 --fractions 0.03`).

On Linux, `run_local.sh` creates the virtualenv, installs dependencies, and
runs everything in one command.
