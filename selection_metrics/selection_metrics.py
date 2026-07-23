# -*- coding: utf-8 -*-
"""
selection_metrics.py
====================

Measure the three theory-linked selection-quality quantities of the
CHV-CRAIG / CHVCoreset pipeline, at *fresh-init* (epoch-0) model state:

    (1) certificate radius        eps_hat_c   = max_{i in I_c} min_{h in H_c} d(g_i, g_h)
        (Prop. `prop:certificate`)             -- covering radius of the CHVS4 pool H_c

    (2) coverage loss             L_c(S_c)    = sum_{i in I_c} min_{j in S_c} d(g_i, g_j)
        (eq. `eq:coverage_loss`)               -- the objective greedy CRAIG minimises

    (3) covering radius of S_c    R_c(S_c)    = max_{i in I_c} min_{j in S_c} d(g_i, g_j)
        (Gonzalez a-priori net)                -- worst-case per-point coverage of the coreset

All three are reported under FOUR distance conventions so the open
"d vs d^2" and "raw z vs normalised g" decisions can be settled from data:

    raw_d    : d(z_i, z_j) = ||z_i - z_j||_2                 (raw Euclidean, what CHVCoreset uses on Z_c)
    raw_d2   : d(z_i, z_j) = ||z_i - z_j||_2^2               (raw squared, what the *code* minimises)
    cos_d    : d(g_i, g_j) = sqrt(2 - 2 g_i^T g_j)           (normalised cosine, manuscript eq. line 853)
    cos_d2   : d(g_i, g_j) = 2 - 2 g_i^T g_j                 (normalised squared)

Run `python selection_metrics.py --selftest` for a NumPy-only invariant check.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

# --------------------------------------------------------------------------- #
#  Section A.  Core, dataset-agnostic metric functions (pure NumPy)           #
# --------------------------------------------------------------------------- #

_EPS = 1e-12


def _as_float64(G: np.ndarray) -> np.ndarray:
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D (n_c x C); got shape {G.shape}")
    return G


def _normalise_rows(G: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation g_i = z_i / ||z_i|| (for the cosine convention)."""
    nrm = np.linalg.norm(G, axis=1, keepdims=True)
    return G / np.maximum(nrm, _EPS)


def _min_dist_to_reference(G_query, G_ref, squared=False, block=2048):
    """Per query row, the min Euclidean (or squared) distance to the ref set."""
    nq = G_query.shape[0]
    q_sq = np.einsum("ij,ij->i", G_query, G_query)
    r_sq = np.einsum("ij,ij->i", G_ref, G_ref)
    out = np.empty(nq, dtype=np.float64)
    for s in range(0, nq, block):
        e = min(s + block, nq)
        cross = G_query[s:e] @ G_ref.T
        d2 = q_sq[s:e, None] - 2.0 * cross + r_sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        out[s:e] = d2.min(axis=1)
    return out if squared else np.sqrt(out)


def certificate_epsilon(G_c, H_local, squared=False):
    """eps_hat_c = max_{i in I_c} min_{h in H_c} d(g_i, g_h)."""
    return float(_min_dist_to_reference(G_c, G_c[H_local], squared=squared).max())


def coverage_loss(G_c, S_local, squared=False):
    """L_c(S_c) = sum_{i in I_c} min_{j in S_c} d(g_i, g_j)."""
    return float(_min_dist_to_reference(G_c, G_c[S_local], squared=squared).sum())


def covering_radius(G_c, S_local, squared=False):
    """R_c(S_c) = max_{i in I_c} min_{j in S_c} d(g_i, g_j)."""
    return float(_min_dist_to_reference(G_c, G_c[S_local], squared=squared).max())


def all_metrics_one_convention(G_c, H_local, S_local, squared):
    """All coverage statistics for one distance convention on given G_c."""
    dS = _min_dist_to_reference(G_c, G_c[S_local], squared=squared)
    dH = _min_dist_to_reference(G_c, G_c[H_local], squared=squared)
    return {
        "eps_hat_c":          float(dH.max()),
        "covering_radius_Sc": float(dS.max()),
        "coverage_loss_Sc":   float(dS.sum()),
        "coverage_loss_Hc":   float(dH.sum()),
        "mean_dist_Sc":       float(dS.mean()),
        "mean_dist_Hc":       float(dH.mean()),
    }


def compute_class_metrics(Z_c, H_local, S_local,
                          conventions=("raw_d", "raw_d2", "cos_d", "cos_d2")):
    """Full metric row for one class under all requested conventions."""
    Z_c = _as_float64(Z_c)
    H_local = np.asarray(H_local, dtype=int)
    S_local = np.asarray(S_local, dtype=int)
    G_c = _normalise_rows(Z_c)
    n_c = Z_c.shape[0]
    row = {
        "n_c": int(n_c),
        "k_H": int(len(np.unique(H_local))),
        "r_c": int(len(np.unique(S_local))),
        "S_subset_H": bool(np.all(np.isin(S_local, H_local))),
        "z_norm_min": float(np.linalg.norm(Z_c, axis=1).min()),
        "z_norm_max": float(np.linalg.norm(Z_c, axis=1).max()),
        "z_norm_mean": float(np.linalg.norm(Z_c, axis=1).mean()),
    }
    conv_spec = {"raw_d": (Z_c, False), "raw_d2": (Z_c, True),
                 "cos_d": (G_c, False), "cos_d2": (G_c, True)}
    for name in conventions:
        base, sq = conv_spec[name]
        for k, v in all_metrics_one_convention(base, H_local, S_local, sq).items():
            row[f"{name}__{k}"] = v
    return row


def check_invariants(row, conventions=("raw_d", "cos_d")):
    """I1 S subset H; I2 eps=R(H)<=R(S); I3 L(S)>=L(H); I4 L(S)<=n_c*R(S); I5 nonneg."""
    checks = [("I1: S_c subset H_c", bool(row["S_subset_H"]), "")]
    tol = 1e-6
    for c in conventions:
        eps = row[f"{c}__eps_hat_c"]; RS = row[f"{c}__covering_radius_Sc"]
        LS = row[f"{c}__coverage_loss_Sc"]; LH = row[f"{c}__coverage_loss_Hc"]
        n_c = row["n_c"]
        checks.append((f"I2[{c}]: eps_hat<=R(S)", eps <= RS + tol, f"{eps:.4g} vs {RS:.4g}"))
        checks.append((f"I3[{c}]: L(S)>=L(H)", LS >= LH - tol, f"{LS:.4g} vs {LH:.4g}"))
        checks.append((f"I4[{c}]: L(S)<=n_c*R(S)", LS <= n_c * RS + tol, f"{LS:.4g} vs {n_c*RS:.4g}"))
        checks.append((f"I5[{c}]: nonneg", min(eps, RS, LS, LH) >= -tol, ""))
    return checks


# --------------------------------------------------------------------------- #
#  Section B.  Pipeline adapters                                              #
# --------------------------------------------------------------------------- #

def chvs4_pool(Z_c, k_H, seed=0):
    """CHVS4 candidate pool H_c (repo wrapper + self-contained fallback)."""
    n_c = Z_c.shape[0]
    k_H = int(min(k_H, n_c))
    local_ids = list(range(n_c))
    try:
        import random
        random.seed(seed)
        from algorithms import find_convex_hull_vertices
        picked = find_convex_hull_vertices(Z_c, local_ids, max_vertices=k_H)
        return np.asarray(sorted(set(int(i) for i in picked)), dtype=int)
    except Exception as exc:
        print(f"   [chvs4_pool] repo CHVS4 unavailable ({exc}); using fallback.")
        return _chvs4_fallback(Z_c, k_H, seed)


def _chvs4_fallback(Z_c, k_H, seed=0):
    """Self-contained CHVS4-style furthest-from-affine-hull vertex picker."""
    rng = np.random.default_rng(seed)
    n, d = Z_c.shape
    if n <= d + 1:
        return np.arange(n)
    x0_idx = int(rng.integers(n)); x0 = Z_c[x0_idx]
    first = int(np.argmax(np.linalg.norm(Z_c - x0, axis=1)))
    S = {x0_idx, first}
    while len(S) < k_H:
        cur = list(S)
        basis = (Z_c[cur] - x0).T
        avail = np.array(sorted(set(range(n)) - S), dtype=int)
        if avail.size == 0:
            break
        if avail.size > 2000:
            avail = rng.choice(avail, size=2000, replace=False)
        V = Z_c[avail] - x0
        Qi = np.linalg.pinv(basis.T @ basis)
        res = np.linalg.norm(V - (V @ basis @ Qi) @ basis.T, axis=1)
        b = int(np.argmax(res))
        if res[b] <= 1e-4:
            break
        S.add(int(avail[b]))
    return np.asarray(sorted(S), dtype=int)


def greedy_craig(Z_c, pool_local, r_c, squared=True):
    """Greedy facility-location coreset S_c subset H_c of size r_c.
    squared=True => d^2 (current code); squared=False => d (lemmas)."""
    pool_local = np.asarray(pool_local, dtype=int)
    r_c = int(min(r_c, len(pool_local)))
    G_pool = Z_c[pool_local]
    z_sq = np.einsum("ij,ij->i", Z_c, Z_c)
    p_sq = np.einsum("ij,ij->i", G_pool, G_pool)
    D2 = z_sq[:, None] - 2.0 * (Z_c @ G_pool.T) + p_sq[None, :]
    np.maximum(D2, 0.0, out=D2)
    D = D2 if squared else np.sqrt(D2)
    n_c = Z_c.shape[0]
    min_d = np.full(n_c, np.inf)
    chosen, remaining = [], list(range(len(pool_local)))
    for _ in range(r_c):
        best_gain, best_pos = -np.inf, -1
        for pos in remaining:
            gain = np.maximum(0.0, min_d - D[:, pos]).sum()
            if gain > best_gain:
                best_gain, best_pos = gain, pos
        if best_pos < 0:
            break
        chosen.append(best_pos); remaining.remove(best_pos)
        min_d = np.minimum(min_d, D[:, best_pos])
    return pool_local[np.asarray(chosen, dtype=int)]


def ff_augment(Z_c, pool_local, k_H, squared=False):
    """Farthest-first (Gonzalez) augmentation of the hull pool up to k_H.

    In the (C-1)-dimensional logit-gradient space the convex-hull phase
    saturates at ~C affinely independent vertices, so it cannot by itself
    fill a candidate budget k_H = alpha * r_c > C. This routine implements the
    farthest-first fill of Algorithm 1 (see Theorem thm:apriori): starting
    from the hull vertices, it repeatedly adds the class point that is
    farthest from the current pool, which controls the covering radius.
    Returns LOCAL indices into Z_c (a superset of pool_local, size <= k_H).
    """
    pool = list(dict.fromkeys(int(i) for i in np.asarray(pool_local, dtype=int)))
    k_H = int(min(k_H, Z_c.shape[0]))
    if len(pool) >= k_H:
        return np.asarray(sorted(pool[:k_H]), dtype=int)
    d = _min_dist_to_reference(Z_c, Z_c[np.asarray(pool, dtype=int)], squared=squared)
    d[np.asarray(pool, dtype=int)] = -1.0
    while len(pool) < k_H:
        j = int(np.argmax(d))
        if d[j] <= 0:
            break
        pool.append(j)
        dj = _min_dist_to_reference(Z_c, Z_c[j:j + 1], squared=squared)
        np.minimum(d, dj, out=d)
        d[j] = -1.0
    return np.asarray(sorted(pool), dtype=int)


def build_pool(Z_c, k_H, seed=0, augment=True, squared=False):
    """Candidate pool H_c: CHVS4 hull vertices, then (default) farthest-first
    augmentation up to k_H = alpha * r_c."""
    H = chvs4_pool(Z_c, k_H, seed=seed)
    if augment and len(H) < min(k_H, Z_c.shape[0]):
        H = ff_augment(Z_c, H, k_H, squared=squared)
    return H



def extract_logit_gradients(model, dataset, num_classes, device="cpu", batch_size=256):
    """z_i = softmax(logits) - onehot(y_i) for every sample (CRAIG.py logic)."""
    import torch
    import torch.nn.functional as F
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    Z, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            probs = F.softmax(model(x), dim=1)
            onehot = F.one_hot(y, num_classes=num_classes).float()
            Z.append((probs - onehot).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(Z, axis=0), np.concatenate(ys, axis=0)


# --------------------------------------------------------------------------- #
#  Section C.  Orchestration                                                   #
# --------------------------------------------------------------------------- #

def measure_from_embeddings(Z, labels, fraction, alpha, seed,
                            dataset="", model="", squared_selection=True,
                            pool_augment=True,
                            conventions=("raw_d", "raw_d2", "cos_d", "cos_d2")):
    """Per-class pool+coreset selection and metrics. r_c proportional; k_H=alpha*r_c."""
    Z = _as_float64(Z); labels = np.asarray(labels)
    N = Z.shape[0]; budget = int(round(N * fraction))
    rows = []
    for c in np.unique(labels):
        I_c = np.where(labels == c)[0]; n_c = len(I_c)
        if n_c == 0:
            continue
        Z_c = Z[I_c]
        r_c = max(1, min(int(round(budget * (n_c / N))), n_c))
        k_H = int(min(max(int(round(alpha * r_c)), r_c), n_c))
        t0 = time.time(); H_local = build_pool(Z_c, k_H, seed=seed, augment=pool_augment); t_pool = time.time() - t0
        t0 = time.time(); S_local = greedy_craig(Z_c, H_local, r_c, squared=squared_selection); t_greedy = time.time() - t0
        row = {"dataset": dataset, "model": model, "method": "CHV-CRAIG",
               "fraction": fraction, "alpha": alpha, "seed": seed, "class": int(c),
               "T_pool_s": round(t_pool, 4), "T_greedy_s": round(t_greedy, 4)}
        row.update(compute_class_metrics(Z_c, H_local, S_local, conventions=conventions))
        rows.append(row)
    return rows


def aggregate_rows(rows, conventions=("raw_d", "raw_d2", "cos_d", "cos_d2")):
    """Class-size-weighted aggregate for one (dataset,frac,seed)."""
    if not rows:
        return {}
    N = sum(r["n_c"] for r in rows)
    agg = {k: rows[0][k] for k in ("dataset", "model", "method", "fraction", "alpha", "seed")}
    agg["N"] = N; agg["num_classes"] = len(rows)
    agg["k_H_total"] = sum(r["k_H"] for r in rows); agg["r_c_total"] = sum(r["r_c"] for r in rows)
    for c in conventions:
        agg[f"{c}__eps_hat_max"] = max(r[f"{c}__eps_hat_c"] for r in rows)
        agg[f"{c}__eps_hat_wmean"] = sum(r[f"{c}__eps_hat_c"] * r["n_c"] for r in rows) / N
        agg[f"{c}__covering_radius_max"] = max(r[f"{c}__covering_radius_Sc"] for r in rows)
        agg[f"{c}__coverage_loss_total"] = sum(r[f"{c}__coverage_loss_Sc"] for r in rows)
        agg[f"{c}__coverage_loss_pool_total"] = sum(r[f"{c}__coverage_loss_Hc"] for r in rows)
        agg[f"{c}__mean_dist_wmean"] = sum(r[f"{c}__coverage_loss_Sc"] for r in rows) / N
    return agg


def write_csv(rows, path):
    import csv
    if not rows:
        print(f"   [write_csv] no rows for {path}"); return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"   [write_csv] wrote {len(rows)} rows -> {path}")


# ---- dataset builders (fresh-init model) ---------------------------------- #

def build_fashionmnist(device="cpu"):
    import torch, torch.nn as nn, torchvision
    import torchvision.transforms as T
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
    ds = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)

    class LogReg(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    return LogReg().to(device), ds, 10


def _bundled_resnet20():
    """Self-contained ResNet-20 (CIFAR); same architecture as CRAIG.py."""
    import torch.nn as nn
    import torch.nn.functional as F

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out)); out += self.shortcut(x)
            return F.relu(out)

    class ResNet(nn.Module):
        def __init__(self, num_blocks=(3, 3, 3), num_classes=10):
            super().__init__()
            self.in_planes = 16
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = self._make(16, num_blocks[0], 1)
            self.layer2 = self._make(32, num_blocks[1], 2)
            self.layer3 = self._make(64, num_blocks[2], 2)
            self.linear = nn.Linear(64, num_classes)

        def _make(self, planes, n, stride):
            layers, strides = [], [stride] + [1] * (n - 1)
            for s in strides:
                layers.append(BasicBlock(self.in_planes, planes, s)); self.in_planes = planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer3(self.layer2(self.layer1(out)))
            out = F.avg_pool2d(out, out.size()[3]).view(out.size(0), -1)
            return self.linear(out)

    return ResNet()


def build_cifar10_resnet20(device="cpu"):
    import torch, torchvision
    import torchvision.transforms as T
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from CRAIG import resnet20
        model = resnet20()
    except Exception:
        model = _bundled_resnet20()
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    return model.to(device), ds, 10


def build_covtype(device="cpu"):
    """Covtype tabular logistic regression (7 classes)."""
    import torch, torch.nn as nn
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler
    data = fetch_covtype()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = (data.target - 1).astype(np.int64)
    n_features, n_classes = X.shape[1], 7

    class TabDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(y)

        def __getitem__(self, i):
            return torch.from_numpy(X[i]), int(y[i])

    class LogReg(nn.Module):
        def __init__(self):
            super().__init__(); self.fc = nn.Linear(n_features, n_classes)

        def forward(self, x):
            return self.fc(x)

    return LogReg().to(device), TabDS(), n_classes


DATASET_BUILDERS = {"fashionmnist": build_fashionmnist,
                    "cifar10": build_cifar10_resnet20,
                    "covtype": build_covtype}


def run_dataset(name, fractions, seeds, alpha, outdir, device="cpu", squared_selection=True, pool_augment=True):
    import torch
    builder = DATASET_BUILDERS[name]
    per_class_rows, summary_rows = [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model, ds, num_classes = builder(device=device)
        print(f"[{name}] seed={seed}: extracting fresh-init logit gradients "
              f"({len(ds)} samples, {num_classes} classes)...")
        Z, labels = extract_logit_gradients(model, ds, num_classes, device=device)
        for frac in fractions:
            rows = measure_from_embeddings(Z, labels, frac, alpha, seed,
                                           dataset=name, model=type(model).__name__,
                                           squared_selection=squared_selection,
                                           pool_augment=pool_augment)
            per_class_rows.extend(rows); summary_rows.append(aggregate_rows(rows))
            bad = []
            for r in rows:
                for cname, ok, detail in check_invariants(r):
                    if not ok:
                        bad.append((r["class"], cname, detail))
            tag = "OK" if not bad else f"VIOLATIONS={bad[:3]}"
            print(f"   frac={frac:<5} classes={len(rows)}  invariants:{tag}")
    write_csv(per_class_rows, os.path.join(outdir, f"selection_metrics_perclass_{name}.csv"))
    write_csv(summary_rows, os.path.join(outdir, f"selection_metrics_summary_{name}.csv"))
    return per_class_rows, summary_rows


# --------------------------------------------------------------------------- #
#  Section D.  Synthetic self-test (NumPy only)                               #
# --------------------------------------------------------------------------- #

def _synthetic_zero_sum_class(n_c, C, n_modes=6, spread=0.8, seed=0):
    rng = np.random.default_rng(seed)
    modes = rng.normal(size=(n_modes, C))
    modes -= modes.mean(axis=1, keepdims=True)
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    which = rng.integers(0, n_modes, size=n_c)
    radii = rng.beta(1.5, 3.0, size=n_c)
    Z = radii[:, None] * modes[which] * np.sqrt(2.0)
    Z += spread * 0.15 * rng.normal(size=(n_c, C))
    Z -= Z.mean(axis=1, keepdims=True)
    return Z


def selftest(seed=0):
    print("=" * 68)
    print("SYNTHETIC SELF-TEST  (NumPy only; verifies theory invariants)")
    print("=" * 68)
    all_ok = True
    for (n_c, C, r_c, alpha) in [(1000, 10, 30, 5), (3000, 20, 50, 5),
                                 (500, 7, 20, 4), (2000, 100, 40, 3)]:
        Z_c = _synthetic_zero_sum_class(n_c, C, seed=seed)
        H = build_pool(Z_c, min(alpha * r_c, n_c), seed=seed, augment=True)
        for sq in (True, False):
            S = greedy_craig(Z_c, H, r_c, squared=sq)
            row = compute_class_metrics(Z_c, H, S)
            checks = check_invariants(row, conventions=("raw_d", "raw_d2", "cos_d", "cos_d2"))
            ok = all(c[1] for c in checks); all_ok &= ok
            sel = "d^2" if sq else "d"
            print(f"\n n_c={n_c:<5} C={C:<4} r_c={r_c} k_H={len(H)} greedy-on={sel}:  "
                  f"{'ALL PASS' if ok else 'FAIL'}")
            print(f"   eps_hat_c(raw_d)={row['raw_d__eps_hat_c']:.4f}  "
                  f"R(S,raw_d)={row['raw_d__covering_radius_Sc']:.4f}  "
                  f"L(S,raw_d)={row['raw_d__coverage_loss_Sc']:.2f}")
            print(f"   ||z|| in [{row['z_norm_min']:.3f}, {row['z_norm_max']:.3f}], "
                  f"max<=sqrt2={np.sqrt(2):.3f}: "
                  f"{'ok' if row['z_norm_max'] <= np.sqrt(2)+1e-6 else 'CHECK'}")
            for name, cok, detail in checks:
                if not cok:
                    print(f"      [FAIL] {name}  {detail}")
    print("\n" + "=" * 68)
    print("RESULT:", "[OK] ALL INVARIANTS HOLD" if all_ok else "[FAIL] SOME INVARIANTS FAILED")
    print("=" * 68)
    return all_ok


# --------------------------------------------------------------------------- #
#  Section E.  CLI                                                             #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Measure eps_hat_c, coverage loss and covering radius "
                    "of the CHV-CRAIG pipeline at fresh-init model state.")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--datasets", nargs="+", default=["fashionmnist"],
                   choices=list(DATASET_BUILDERS.keys()))
    p.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.1])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--alpha", type=float, default=5.0, help="k_H = alpha * r_c")
    p.add_argument("--outdir", default="./selection_metrics_out")
    p.add_argument("--device", default="cpu")
    p.add_argument("--selection-distance", choices=["d", "d2"], default="d2")
    p.add_argument("--no-pool-augment", action="store_true",
                   help="disable farthest-first pool fill (hull-only pool, k_H may be < alpha*r_c)")
    args = p.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    squared = (args.selection_distance == "d2")
    os.makedirs(args.outdir, exist_ok=True)
    for name in args.datasets:
        run_dataset(name, args.fractions, args.seeds, args.alpha,
                    args.outdir, device=args.device, squared_selection=squared,
                    pool_augment=not args.no_pool_augment)
    print("\nDone. CSVs in", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
