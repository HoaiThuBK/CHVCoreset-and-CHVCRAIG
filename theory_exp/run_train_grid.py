# -*- coding: utf-8 -*-
"""
run_train_grid.py  --  Section 3.2 accuracy/runtime ablation over (alpha, Delta)
===============================================================================

Drives the EXISTING training scripts (`train_logreg_fmnist.py`,
`train_resnet20_cifar10.py`, `train_resnet20_svhn.py`) over a grid of the
candidate multiplier alpha (`--candidate_multiplier`) and the re-selection
period Delta (`--update_freq`), then collects, for each run, the final test
accuracy and the cumulative selection / training times.

Because the train scripts do not encode alpha/Delta in their output filenames,
this driver reads and archives each produced CSV immediately after the run.

Example (reduce --epochs for a fast ablation):
    python run_train_grid.py --dataset fashionmnist \
        --alphas 3 5 10 --deltas 40 --fractions 0.03 --seeds 42 43 44 \
        --epochs 100 --warmup 10 --device cuda --code_dir ../chv_craig \
        --outcsv out_ablation/train_grid_fashionmnist.csv

Output: one aggregated CSV row per (alpha, Delta, fraction, seed) with columns
    dataset, method, alpha, delta, fraction, seed, final_acc, T_sel, T_train, T_tot
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

SPEC = {
    # dataset: (script, results_dir, filename_prefix)
    "fashionmnist": ("train_logreg_fmnist.py", "results_mnist", "results_mnist_logreg"),
    "cifar10":      ("train_resnet20_cifar10.py", "results_cifar10", "results_cifar10_resnet"),
    "svhn":         ("train_resnet20_svhn.py", "results_svhn", "results_svhn_resnet20"),
}


def expected_csv(code_dir, dataset, method, gtype, seed, frac):
    _, rdir, prefix = SPEC[dataset]
    fn = f"{prefix}_{method}_{gtype}_seed{seed}"
    if method != "full_dataset":
        fn += f"_frac{frac}"
    fn += ".csv"
    return os.path.join(code_dir, rdir, method, fn)


def collect(csv_path):
    """Return (final_acc, T_sel_cumulative, T_train_cumulative)."""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return None
    T_sel = sum(float(r.get("selection_time_s", 0) or 0) for r in rows)
    T_train = sum(float(r.get("train_time_s", 0) or 0) for r in rows)
    final_acc = float(rows[-1]["accuracy"])
    return final_acc, T_sel, T_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPEC.keys()))
    ap.add_argument("--method", default="chvcoreset",
                    choices=["chv_craig", "chvcoreset", "craig"])
    ap.add_argument("--alphas", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--deltas", nargs="+", type=int, default=[40])
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.03])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--gradient_type", default="logit")
    ap.add_argument("--device", default="cuda")   # forwarded via CUDA_VISIBLE_DEVICES only
    ap.add_argument("--code_dir", default="../src")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--outcsv", default="out_ablation/train_grid.csv")
    ap.add_argument("--archive_dir", default="out_ablation/train_grid_runs")
    args = ap.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    script = os.path.join(code_dir, SPEC[args.dataset][0])
    assert os.path.isfile(script), f"train script not found: {script}"
    os.makedirs(os.path.dirname(os.path.abspath(args.outcsv)), exist_ok=True)
    os.makedirs(args.archive_dir, exist_ok=True)

    results = []
    for a in args.alphas:
        for d in args.deltas:
            for frac in args.fractions:
                for seed in args.seeds:
                    cmd = [args.python, script,
                           "--selection_method", args.method,
                           "--coreset_fraction", str(frac),
                           "--candidate_multiplier", str(a),
                           "--update_freq", str(d),
                           "--seed", str(seed),
                           "--gradient_type", args.gradient_type]
                    if args.epochs is not None:
                        cmd += ["--epochs", str(args.epochs)]
                    if args.warmup is not None:
                        cmd += ["--warmup_epochs", str(args.warmup)]
                    print("\n>>>", " ".join(cmd))
                    t0 = time.time()
                    subprocess.run(cmd, cwd=code_dir, check=True)
                    wall = time.time() - t0
                    cpath = expected_csv(code_dir, args.dataset, args.method,
                                         args.gradient_type, seed, frac)
                    if not os.path.isfile(cpath):
                        hits = glob.glob(os.path.join(code_dir, SPEC[args.dataset][1],
                                                      args.method, "*.csv"))
                        cpath = max(hits, key=os.path.getmtime) if hits else None
                    if not cpath:
                        print("   [warn] output CSV not found; skipping"); continue
                    got = collect(cpath)
                    if got is None:
                        continue
                    acc, T_sel, T_train = got
                    row = {"dataset": args.dataset, "method": args.method,
                           "alpha": a, "delta": d, "fraction": frac, "seed": seed,
                           "final_acc": round(acc, 3),
                           "T_sel": round(T_sel, 2), "T_train": round(T_train, 2),
                           "T_tot": round(T_sel + T_train, 2),
                           "wall_s": round(wall, 1)}
                    results.append(row)
                    # archive the run's CSV with alpha/delta in the name
                    dst = os.path.join(
                        args.archive_dir,
                        f"{args.dataset}_{args.method}_a{a}_d{d}_frac{frac}_seed{seed}.csv")
                    try:
                        os.replace(cpath, dst)
                    except Exception:
                        pass
                    # incremental write (so partial grids are not lost)
                    with open(args.outcsv, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
                        w.writeheader()
                        w.writerows(results)
                    print(f"   acc={acc:.2f}  T_sel={T_sel:.1f}s  T_tot={T_sel+T_train:.1f}s")
    print("\nDone. grid CSV ->", os.path.abspath(args.outcsv))


if __name__ == "__main__":
    main()
