# -*- coding: utf-8 -*-
"""
measure_theory.py  --  Experiments for manuscript section 3.1
=============================================================

Measures, with the ACTUAL CHVS4 pipeline, the theory-linked quantities that
connect the analysis to the experiments:

    * eps_hat_c        computable a posteriori certificate (Prop. certificate)
    * coverage loss    L_c(S_c) for CHV-CRAIG, compared with plain CRAIG
    * gradient error   relative ||sum z - sum gamma z|| / ||sum z||
                       (Lemma "coverage loss controls the logit-gradient error"
                        + End-to-end corollary), for CHV-CRAIG vs CRAIG
    * selection times  T_pool (CHVS4) and T_greedy

Selection uses the real pipeline: CHVS4_Algorithm3_Ding2017 for the candidate
pool H_c (budget alpha*r_c), craig_greedy_cdist for the coreset S_c, and
calculate_weights for the count weights -- so numbers are faithful to the paper.

Usage (on a machine with torch/torchvision + the datasets):
    python measure_theory.py --datasets fashionmnist cifar10 svhn \
        --fractions 0.01 0.03 0.05 --seeds 42 43 44 --alpha 5 \
        --warmup fashionmnist:10 cifar10:20 svhn:10 --device cuda \
        --outdir out_theory

Outputs:
    out_theory/theory_perclass_<dataset>.csv     (one row per class/frac/seed)
    out_theory/theory_summary_<dataset>.csv       (aggregated mean/std over seeds)
"""
from __future__ import annotations
import argparse
import csv
import os
import time
import numpy as np

import theory_common as tc


def parse_warmup(items, default):
    """--warmup fashionmnist:10 cifar10:20 svhn:10  ->  dict."""
    d = dict(default)
    for it in items or []:
        k, v = it.split(":")
        d[k.lower()] = int(v)
    return d


def farthest_first_pool(Znp, k, rng):
    """Gonzalez farthest-first pool of size k (numpy, O(n*k))."""
    n = Znp.shape[0]
    start = int(rng.integers(n))
    chosen = [start]
    dmin = np.linalg.norm(Znp - Znp[start], axis=1)
    while len(chosen) < min(k, n):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(Znp - Znp[j], axis=1))
    return chosen


def measure_dataset(dataset, fractions, seeds, alpha, warmup_epochs, device,
                    code_dir, data_root, selection_no_aug, chvs4_epsilon=0.0,
                    pool_method="chvs4"):
    import torch
    from utils import get_indices_by_class, compute_gradient_representations, calculate_weights
    from chvs4 import CHVS4_Algorithm3_Ding2017
    from craig import craig_greedy_cdist

    per_class_rows, summary_rows = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

        sel_ds, _train_full, testset, model, num_classes = tc.build_dataset_and_model(
            dataset, device, data_root=data_root, selection_no_aug=selection_no_aug)
        print(f"[{dataset}] seed={seed}: warming up {warmup_epochs} epochs ...")
        tc.warmup_model(model, sel_ds, testset, warmup_epochs, device)

        indices_by_class = get_indices_by_class(sel_ds, num_classes)
        N = sum(len(v) for v in indices_by_class.values())

        # cache per-class logit gradients once (model is fixed after warmup)
        Zc_cache = {}
        for c, cidx in indices_by_class.items():
            if len(cidx) == 0:
                continue
            g = compute_gradient_representations(
                model=model, full_dataset=sel_ds, indices=cidx, device=device,
                gradient_type="logit", batch_size=256)
            Zc_cache[c] = (cidx, g.detach().cpu().numpy().astype(np.float64), g)

        for f in fractions:
            total_size = max(1, int(N * f))
            alloc = tc.allocate_budgets(indices_by_class, total_size)

            # accumulators for the GLOBAL relative gradient-matching error
            Gfull = None; Gchv = None; Gcraig = None
            agg = {"eps_max": 0.0, "eps_wsum": 0.0,
                   "Lcov_chv": 0.0, "Lcov_craig": 0.0,
                   "covrad_chv_max": 0.0,
                   "kH_total": 0, "rc_total": 0, "n": 0,
                   "t_pool": 0.0, "t_greedy": 0.0, "t_craig": 0.0}

            for c, (cidx, Znp, Zt) in Zc_cache.items():
                r_c = int(alloc.get(c, 0))
                n_c = Znp.shape[0]
                if r_c <= 0 or n_c == 0:
                    continue
                r_c = min(r_c, n_c)
                k_H = int(min(n_c, max(r_c, alpha * r_c)))

                # ---- pool generation (chvs4 = real pipeline; random/ff = M1 ablation) ----
                t0 = time.time()
                if pool_method == "random":
                    rng = np.random.default_rng(seed * 100003 + int(c))
                    H_local = rng.choice(n_c, size=k_H, replace=False).astype(int).tolist()
                elif pool_method == "ff":
                    rng = np.random.default_rng(seed * 100003 + int(c))
                    H_local = farthest_first_pool(Znp, k_H, rng)
                else:  # "chvs4"
                    H_local = CHVS4_Algorithm3_Ding2017(
                        Znp, epsilon=chvs4_epsilon, budget=k_H, random_state=seed
                    ).astype(int).tolist()
                H_local = list(dict.fromkeys(H_local))
                t_pool = time.time() - t0

                t0 = time.time()
                if len(H_local) <= r_c:
                    S_local = list(H_local)
                else:
                    S_local = craig_greedy_cdist(Zt, budget=r_c,
                                                 candidate_indices_local=H_local,
                                                 desc=f"chv {dataset} c{c}")
                t_greedy = time.time() - t0
                w_chv = calculate_weights(Zt, S_local)

                # ---- plain CRAIG on the full class (baseline) ----
                t0 = time.time()
                S_craig = craig_greedy_cdist(Zt, budget=r_c,
                                             candidate_indices_local=None,
                                             desc=f"craig {dataset} c{c}")
                t_craig = time.time() - t0
                w_craig = calculate_weights(Zt, S_craig)

                # ---- metrics (Euclidean d) ----
                eps_c = tc.eps_hat(Znp, H_local)
                Lchv, Rchv = tc.coverage_and_radius(Znp, S_local)
                Lcraig, _ = tc.coverage_and_radius(Znp, S_craig)

                gfull_c = Znp.sum(axis=0)
                gchv_c = tc.weighted_class_sum(Znp, w_chv)
                gcraig_c = tc.weighted_class_sum(Znp, w_craig)
                Gfull = gfull_c if Gfull is None else Gfull + gfull_c
                Gchv = gchv_c if Gchv is None else Gchv + gchv_c
                Gcraig = gcraig_c if Gcraig is None else Gcraig + gcraig_c

                per_class_rows.append({
                    "dataset": dataset, "pool": pool_method,
                    "seed": seed, "fraction": f, "class": int(c),
                    "n_c": int(n_c), "r_c": int(r_c), "k_H": int(len(H_local)),
                    "eps_hat_c": round(eps_c, 6),
                    "cover_loss_chv": round(Lchv, 4),
                    "cover_loss_craig": round(Lcraig, 4),
                    "covering_radius_chv": round(Rchv, 6),
                    "grad_err_chv_class": round(float(np.linalg.norm(gfull_c - gchv_c) /
                                                      (np.linalg.norm(gfull_c) + 1e-12)), 6),
                    "grad_err_craig_class": round(float(np.linalg.norm(gfull_c - gcraig_c) /
                                                        (np.linalg.norm(gfull_c) + 1e-12)), 6),
                    "T_pool_s": round(t_pool, 4), "T_greedy_s": round(t_greedy, 4),
                    "T_craig_s": round(t_craig, 4),
                })

                agg["eps_max"] = max(agg["eps_max"], eps_c)
                agg["eps_wsum"] += eps_c * n_c
                agg["Lcov_chv"] += Lchv
                agg["Lcov_craig"] += Lcraig
                agg["covrad_chv_max"] = max(agg["covrad_chv_max"], Rchv)
                agg["kH_total"] += len(H_local); agg["rc_total"] += r_c
                agg["n"] += n_c
                agg["t_pool"] += t_pool; agg["t_greedy"] += t_greedy
                agg["t_craig"] += t_craig

            if Gfull is None:
                continue
            gerr_chv = float(np.linalg.norm(Gfull - Gchv) / (np.linalg.norm(Gfull) + 1e-12))
            gerr_craig = float(np.linalg.norm(Gfull - Gcraig) / (np.linalg.norm(Gfull) + 1e-12))
            summary_rows.append({
                "dataset": dataset, "pool": pool_method,
                "seed": seed, "fraction": f, "alpha": alpha,
                "N": agg["n"], "kH_total": agg["kH_total"], "rc_total": agg["rc_total"],
                "eps_hat_max": round(agg["eps_max"], 6),
                "eps_hat_wmean": round(agg["eps_wsum"] / max(1, agg["n"]), 6),
                "cover_loss_chv": round(agg["Lcov_chv"], 3),
                "cover_loss_craig": round(agg["Lcov_craig"], 3),
                "cover_loss_ratio": round(agg["Lcov_chv"] / max(1e-12, agg["Lcov_craig"]), 4),
                "covering_radius_max": round(agg["covrad_chv_max"], 6),
                "grad_err_chv": round(gerr_chv, 6),
                "grad_err_craig": round(gerr_craig, 6),
                "T_pool_s": round(agg["t_pool"], 3),
                "T_greedy_s": round(agg["t_greedy"], 3),
                "T_craig_s": round(agg["t_craig"], 3),
            })
            print(f"   frac={f}: eps_hat_max={agg['eps_max']:.4f} "
                  f"grad_err chv={gerr_chv:.4f} craig={gerr_craig:.4f} "
                  f"Lcov chv/craig={agg['Lcov_chv']/max(1e-12,agg['Lcov_craig']):.3f}")
    return per_class_rows, summary_rows


def write_csv(rows, path):
    if not rows:
        print("   [write_csv] no rows for", path); return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print("   wrote", len(rows), "rows ->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["fashionmnist", "cifar10", "svhn"])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.03, 0.05])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--warmup", nargs="*", default=None,
                    help="per-dataset warmup epochs, e.g. fashionmnist:10 cifar10:20 svhn:10")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--code_dir", default=None, help="path to the main pipeline folder (chv_craig)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--outdir", default="out_theory")
    ap.add_argument("--pool", default="chvs4", choices=["chvs4", "random", "ff"],
                    help="candidate-pool generator: chvs4 (real pipeline), "
                         "random (M1 ablation), ff (farthest-first, M1 ablation)")
    ap.add_argument("--select-on-augmented", action="store_true",
                    help="select on the augmented trainset exactly like the train scripts "
                         "(default: use a non-augmented copy for deterministic gradients)")
    args = ap.parse_args()

    tc.ensure_pipeline_on_path(args.code_dir)
    warm = parse_warmup(args.warmup, {"fashionmnist": 10, "cifar10": 20, "svhn": 10})

    for ds in args.datasets:
        pc, sm = measure_dataset(
            ds, args.fractions, args.seeds, args.alpha,
            warm.get(ds.lower(), 10), args.device, args.code_dir, args.data_root,
            selection_no_aug=not args.select_on_augmented,
            pool_method=args.pool)
        suffix = "" if args.pool == "chvs4" else f"_{args.pool}"
        write_csv(pc, os.path.join(args.outdir, f"theory_perclass_{ds}{suffix}.csv"))
        write_csv(sm, os.path.join(args.outdir, f"theory_summary_{ds}{suffix}.csv"))
    print("\nDone. CSVs in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
