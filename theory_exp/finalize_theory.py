# -*- coding: utf-8 -*-
"""
finalize_theory.py -- one command to turn the theory_summary_*.csv files into
(1) the EXACT LaTeX table used in the manuscript (tab:theory_verify, multirow
    + \\hline, mean+/-std when >=2 seeds, plain value when a single seed),
(2) the two-panel figure fig_theory_verify.pdf (with error bars), and
(3) the prose numbers (Spearman correlations, coverage-ratio ranges).

Run it AFTER measure_theory.py has produced 3-seed CSVs for all datasets:

    # 1) run the missing seeds for CIFAR-10 and SVHN (all three seeds, because
    #    measure_theory.py OVERWRITES the per-dataset summary):
    python measure_theory.py --datasets cifar10 svhn \
        --fractions 0.01 0.03 0.05 --seeds 42 43 44 --alpha 5 \
        --warmup fashionmnist:10 cifar10:20 svhn:10 --device cuda \
        --outdir out_theory

    # 2) regenerate table + figure + prose:
    python finalize_theory.py --indir out_theory \
        --figpath ../../CRAIG_CH_manuscript/fig_theory_verify.pdf

Only numpy/pandas/scipy/matplotlib are needed (no torch).
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS_ORDER = ["fashionmnist", "cifar10", "svhn"]
DS_NAME = {"fashionmnist": "FashionMNIST", "cifar10": "CIFAR-10", "svhn": "SVHN"}
DS_COL = {"fashionmnist": "#1f77b4", "cifar10": "#d62728", "svhn": "#2ca02c"}
MK = {0.01: "o", 0.03: "s", 0.05: "^"}


def load(indir):
    rows = []
    for ds in DS_ORDER:
        p = os.path.join(indir, f"theory_summary_{ds}.csv")
        if os.path.exists(p):
            rows.append(pd.read_csv(p))
    if not rows:
        raise SystemExit(f"No theory_summary_*.csv found in {indir}")
    return pd.concat(rows, ignore_index=True)


def aggregate(df):
    out = []
    for (ds, f), g in df.groupby(["dataset", "fraction"]):
        out.append(dict(
            ds=ds, f=float(f), nseed=len(g),
            eps=g.eps_hat_wmean.mean(), eps_s=g.eps_hat_wmean.std(ddof=0),
            ratio=g.cover_loss_ratio.mean(), ratio_s=g.cover_loss_ratio.std(ddof=0),
            gchv=g.grad_err_chv.mean(), gchv_s=g.grad_err_chv.std(ddof=0),
            gcr=g.grad_err_craig.mean(), gcr_s=g.grad_err_craig.std(ddof=0)))
    R = pd.DataFrame(out)
    R["dsrank"] = R.ds.map({d: i for i, d in enumerate(DS_ORDER)})
    return R.sort_values(["dsrank", "f"]).reset_index(drop=True)


def fmt(m, s, nseed, dec):
    return f"${m:.{dec}f}\\pm{s:.{dec}f}$" if nseed >= 2 else f"${m:.{dec}f}$"


def latex_table(R):
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Theory-linked quantities measured with the CHVS4 pipeline "
             r"across three datasets ($\alpha=5$). $\widehat\varepsilon_c$ is the a "
             r"posteriori certificate of Proposition~\ref{prop:certificate} (weighted "
             r"mean over classes); $L_c^{\mathrm{CHV}}/L_c^{\mathrm{CRAIG}}$ is the "
             r"coverage loss of CHV-CRAIG relative to full CRAIG; the gradient error is "
             r"$\|\sum_i z_i-\sum_j\gamma_j z_j\|_2/\|\sum_i z_i\|_2$ "
             r"(Corollary~\ref{cor:end_to_end}). Values are mean$\pm$std over three "
             r"seeds (42/43/44). The certificate orders the datasets by difficulty "
             r"(FashionMNIST $<$ CIFAR-10 $<$ SVHN) and tracks the coverage inflation, "
             r"while the CHV-CRAIG gradient error stays of the same order as full "
             r"CRAIG.}",
             r"\label{tab:theory_verify}",
             r"\begin{tabular}{llcccc}", r"\hline",
             r"Dataset & Frac. & $\widehat\varepsilon_c$ & "
             r"$L_c^{\mathrm{CHV}}/L_c^{\mathrm{CRAIG}}$ & "
             r"grad.\ err.\ (CHV) & grad.\ err.\ (CRAIG) \\", r"\hline"]
    for ds in DS_ORDER:
        sub = R[R.ds == ds]
        if sub.empty:
            continue
        lines.append(r"\multirow{%d}{*}{%s}" % (len(sub), DS_NAME[ds]))
        for _, r in sub.iterrows():
            lines.append(
                f" & ${r.f*100:.0f}\\%$ & {fmt(r.eps, r.eps_s, r.nseed, 3)} & "
                f"{fmt(r.ratio, r.ratio_s, r.nseed, 2)} & "
                f"{fmt(r.gchv, r.gchv_s, r.nseed, 3)} & "
                f"{fmt(r.gcr, r.gcr_s, r.nseed, 3)} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_figure(R, figpath):
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.grid": True, "grid.alpha": .3})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    rho, p = stats.spearmanr(R.eps, R.ratio)
    for _, r in R.iterrows():
        ax[0].errorbar(r.eps, r.ratio, xerr=r.eps_s, yerr=r.ratio_s, fmt=MK[round(r.f, 2)],
                       ms=7, mfc=DS_COL[r.ds], mec="k", ecolor="gray", elinewidth=.8,
                       capsize=2, zorder=3)
    b, a = np.polyfit(R.eps, R.ratio, 1)
    xs = np.linspace(R.eps.min(), R.eps.max(), 50)
    ax[0].plot(xs, b*xs+a, "k--", lw=1, alpha=.6, zorder=1)
    ax[0].set_xlabel(r"certificate $\widehat{\varepsilon}_c$ (weighted mean over classes)")
    ax[0].set_ylabel(r"coverage-loss ratio $L_c^{\mathrm{CHV}}/L_c^{\mathrm{CRAIG}}$")
    ax[0].set_title(rf"(a) Certificate tracks coverage inflation ($\rho$={rho:.2f}, $p$={p:.3f})",
                    fontsize=8.5)
    order = [(ds, f) for ds in DS_ORDER for f in [.01, .03, .05]]
    labels, chv, cr, chv_s, cr_s = [], [], [], [], []
    for ds, f in order:
        row = R[(R.ds == ds) & (np.isclose(R.f, f))]
        if row.empty:
            continue
        r = row.iloc[0]
        labels.append(f"{int(f*100)}%"); chv.append(r.gchv); cr.append(r.gcr)
        chv_s.append(r.gchv_s); cr_s.append(r.gcr_s)
    x = np.arange(len(labels)); w = .38
    ax[1].bar(x-w/2, chv, w, yerr=chv_s, label="CHV-CRAIG", color="#ff7f0e",
              edgecolor="k", lw=.4, error_kw=dict(elinewidth=.7, capsize=2))
    ax[1].bar(x+w/2, cr, w, yerr=cr_s, label="full CRAIG", color="#7f7f7f",
              edgecolor="k", lw=.4, error_kw=dict(elinewidth=.7, capsize=2))
    ax[1].set_ylim(0, max(chv)*1.25)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.5)
    for i, ds in enumerate(["FashionMNIST", "CIFAR-10", "SVHN"]):
        ax[1].text(1+3*i, -max(chv)*0.18, ds, ha="center", fontsize=7.5,
                   fontweight="bold", clip_on=False)
        if i > 0:
            ax[1].axvline(3*i-0.5, color="k", lw=.5, alpha=.3)
    ax[1].set_ylabel("rel. logit-gradient error")
    ax[1].set_title("(b) End-to-end gradient error stays small", fontsize=8.5)
    ax[1].legend(fontsize=7.5, loc="upper center", ncol=2)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=DS_COL[d],
                  markeredgecolor='k', label=DS_NAME[d], ms=7) for d in DS_ORDER]
    leg += [Line2D([0], [0], marker=MK[f], color='w', markerfacecolor='gray',
                   markeredgecolor='k', label=f"{int(f*100)}%", ms=7) for f in [.01, .03, .05]]
    ax[0].legend(handles=leg, fontsize=6.5, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(figpath, bbox_inches="tight")
    print(f"[figure] saved {figpath}")
    return rho, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="out_theory")
    ap.add_argument("--figpath", default="../../CRAIG_CH_manuscript/fig_theory_verify.pdf")
    args = ap.parse_args()
    R = aggregate(load(args.indir))
    print("\n===== aggregated (mean, std, nseed) =====")
    print(R[["ds", "f", "nseed", "eps", "eps_s", "ratio", "ratio_s",
             "gchv", "gchv_s", "gcr", "gcr_s"]].round(4).to_string(index=False))
    print("\n===== paste-ready tab:theory_verify =====\n")
    print(latex_table(R))
    rho_r, p_r = stats.spearmanr(R.eps, R.ratio)
    rho_g, p_g = stats.spearmanr(R.eps, R.gchv)
    rho_fig, p_fig = make_figure(R, args.figpath)
    print("\n===== prose numbers (update the 4 observations) =====")
    print(f"  certificate vs coverage-ratio : Spearman rho={rho_r:.3f}, p={p_r:.3f}")
    print(f"  certificate vs grad-err (CHV) : Spearman rho={rho_g:.3f}, p={p_g:.3f}  (should stay non-signif.)")
    print(f"  max abs. CHV gradient error   : {R.gchv.max():.3f}")
    print(f"  coverage-ratio range per ds   :")
    for ds in DS_ORDER:
        s = R[R.ds == ds]
        if not s.empty:
            print(f"     {DS_NAME[ds]:12s}: {s.ratio.min():.2f}-{s.ratio.max():.2f}")
    print("\nDrop the LaTeX block above into the manuscript in place of the current")
    print("tab:theory_verify; the figure file has been overwritten in place.")


if __name__ == "__main__":
    main()
