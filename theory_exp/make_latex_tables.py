# -*- coding: utf-8 -*-
"""
make_latex_tables.py  --  turn the measurement CSVs into paste-ready LaTeX
==========================================================================

Reads the CSVs produced by measure_theory.py / ablation.py / run_train_grid.py
and prints LaTeX table snippets (mean +/- std over seeds) ready to paste into
the manuscript.

    python make_latex_tables.py --theory out_theory/theory_summary_*.csv
    python make_latex_tables.py --alpha  out_ablation/ablation_alpha_*.csv
    python make_latex_tables.py --grid   out_ablation/train_grid_*.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
from collections import defaultdict
import numpy as np


def _read(paths):
    rows = []
    for pat in paths:
        for p in glob.glob(pat):
            for r in csv.DictReader(open(p)):
                rows.append(r)
    return rows


def _ms(vals):
    a = np.asarray(vals, float)
    return a.mean(), a.std()


def theory_table(paths):
    rows = _read(paths)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], float(r["fraction"]))].append(r)
    print(r"\begin{table}[t]\centering\footnotesize")
    print(r"\caption{\rev{Theory-linked quantities measured with the CHVS4 pipeline "
          r"(mean$\pm$std over seeds). $\widehat\varepsilon_c$ is the a posteriori "
          r"certificate of Proposition~\ref{prop:certificate}; the gradient error is "
          r"$\|\sum z_i-\sum\gamma_j z_j\|/\|\sum z_i\|$ (Corollary~\ref{cor:end_to_end}). "
          r"CHV-CRAIG matches the full-CRAIG gradient approximation while searching only "
          r"$H_c$.}}")
    print(r"\label{tab:theory_verify}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"Dataset & Frac. & $\widehat\varepsilon_c$ & "
          r"$L_c^{\mathrm{CHV}}/L_c^{\mathrm{CRAIG}}$ & "
          r"grad.\ err.\ (CHV) & grad.\ err.\ (CRAIG) \\")
    print(r"\midrule")
    for (ds, f) in sorted(grp, key=lambda k: (k[0], k[1])):
        g = grp[(ds, f)]
        eps_m, eps_s = _ms([float(x["eps_hat_wmean"]) for x in g])
        rat_m, rat_s = _ms([float(x["cover_loss_ratio"]) for x in g])
        ec_m, ec_s = _ms([float(x["grad_err_chv"]) for x in g])
        cr_m, cr_s = _ms([float(x["grad_err_craig"]) for x in g])
        print(f"{ds} & {f*100:.0f}\\% & "
              f"${eps_m:.3f}\\pm{eps_s:.3f}$ & ${rat_m:.2f}\\pm{rat_s:.2f}$ & "
              f"${ec_m:.3f}\\pm{ec_s:.3f}$ & ${cr_m:.3f}\\pm{cr_s:.3f}$ \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")


def alpha_table(paths):
    rows = _read(paths)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], float(r["alpha"]))].append(r)
    print(r"\begin{table}[t]\centering\footnotesize")
    print(r"\caption{\rev{Effect of the candidate multiplier $\alpha$ "
          r"($k_H=\alpha r_c$) on selection quality and cost "
          r"(mean$\pm$std over seeds). Larger $\alpha$ lowers the certificate and "
          r"the gradient error at higher selection time, justifying $\alpha=5$.}}")
    print(r"\label{tab:ablation_alpha}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"$\alpha$ & $\widehat\varepsilon_c$ & grad.\ err.\ (CHV) & "
          r"$L_c^{\mathrm{CHV}}/L_c^{\mathrm{CRAIG}}$ & $k_H$ (tot.) & $T_{\mathrm{pool}}$ (s) \\")
    print(r"\midrule")
    for (ds, a) in sorted(grp, key=lambda k: (k[0], k[1])):
        g = grp[(ds, a)]
        eps_m, eps_s = _ms([float(x["eps_hat_wmean"]) for x in g])
        ge_m, ge_s = _ms([float(x["grad_err_chv"]) for x in g])
        rat_m, rat_s = _ms([float(x["cover_loss_ratio"]) for x in g])
        kh_m, _ = _ms([float(x["kH_total"]) for x in g])
        tp_m, tp_s = _ms([float(x["T_pool_s"]) for x in g])
        print(f"{a:.0f} & ${eps_m:.3f}\\pm{eps_s:.3f}$ & ${ge_m:.3f}\\pm{ge_s:.3f}$ & "
              f"${rat_m:.2f}\\pm{rat_s:.2f}$ & {kh_m:.0f} & ${tp_m:.2f}\\pm{tp_s:.2f}$ \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")
    print("% (grouped by dataset if multiple present; filter CSVs per dataset for one table each)")


def grid_table(paths):
    rows = _read(paths)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], int(float(r["alpha"])), int(float(r["delta"])), float(r["fraction"]))].append(r)
    print(r"% (alpha, Delta) accuracy/runtime ablation")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Dataset & $\alpha$ & $\Delta$ & Frac. & Acc.\ (\%) & $T_{\mathrm{sel}}$ (s) & $T_{\mathrm{tot}}$ (s) \\")
    print(r"\midrule")
    for k in sorted(grp):
        g = grp[k]
        acc_m, acc_s = _ms([float(x["final_acc"]) for x in g])
        ts_m, _ = _ms([float(x["T_sel"]) for x in g])
        tt_m, _ = _ms([float(x["T_tot"]) for x in g])
        ds, a, d, f = k
        print(f"{ds} & {a} & {d} & {f*100:.0f}\\% & ${acc_m:.2f}\\pm{acc_s:.2f}$ & "
              f"{ts_m:.1f} & {tt_m:.1f} \\\\")
    print(r"\bottomrule\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", nargs="*", default=None)
    ap.add_argument("--alpha", nargs="*", default=None)
    ap.add_argument("--grid", nargs="*", default=None)
    args = ap.parse_args()
    if args.theory:
        print("\n%%%%% Section 3.1 theory table %%%%%")
        theory_table(args.theory)
    if args.alpha:
        print("\n%%%%% Section 3.2 alpha ablation %%%%%")
        alpha_table(args.alpha)
    if args.grid:
        print("\n%%%%% Section 3.2 (alpha,Delta) accuracy/runtime %%%%%")
        grid_table(args.grid)
    if not any([args.theory, args.alpha, args.grid]):
        ap.print_help()


if __name__ == "__main__":
    main()
