# -*- coding: utf-8 -*-
"""
ablation.py  --  Experiments for manuscript section 3.2 (alpha ablation, cheap)
==============================================================================

Sweeps the candidate-pool multiplier alpha (k_H = alpha * r_c) and reports the
SELECTION-QUALITY metrics as a function of alpha, WITHOUT full training:

    eps_hat (certificate), coverage loss, relative gradient-matching error,
    candidate-pool size, and selection time.

This directly supports the claim that larger alpha buys a richer pool (smaller
eps_hat / gradient error) at higher selection cost -- i.e. it justifies the
default alpha=5 as a trade-off. Reuses the real CHVS4 pipeline via
measure_theory.

For the alpha- and Delta-vs-ACCURACY ablation (which needs full training), use
`run_train_grid.py`, which drives the existing train scripts.

Usage:
    python ablation.py --dataset fashionmnist --fraction 0.03 \
        --alphas 3 5 10 --seeds 42 43 44 --warmup 10 --device cuda \
        --outdir out_ablation
"""
from __future__ import annotations
import argparse
import os
import theory_common as tc
import measure_theory as mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashionmnist")
    ap.add_argument("--fraction", type=float, default=0.03)
    ap.add_argument("--alphas", nargs="+", type=float, default=[3, 5, 10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--code_dir", default=None)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--outdir", default="out_ablation")
    ap.add_argument("--select-on-augmented", action="store_true")
    args = ap.parse_args()

    tc.ensure_pipeline_on_path(args.code_dir)

    all_summary = []
    for a in args.alphas:
        print(f"\n===== alpha = {a} =====")
        _, sm = mt.measure_dataset(
            args.dataset, [args.fraction], args.seeds, a, args.warmup,
            args.device, args.code_dir, args.data_root,
            selection_no_aug=not args.select_on_augmented)
        all_summary.extend(sm)
    mt.write_csv(all_summary,
                 os.path.join(args.outdir, f"ablation_alpha_{args.dataset}.csv"))
    print("\nDone. alpha-ablation CSV in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
