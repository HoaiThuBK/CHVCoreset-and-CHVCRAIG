# -*- coding: utf-8 -*-
"""
theory_common.py
================

Shared utilities for the theory-verification experiments (§3.1) and the
alpha/Delta ablation (§3.2) of the CHVCoreset / CHV-CRAIG paper.

Everything here reuses the ACTUAL experiment pipeline in `CRAIG-CH-main`
(the same `chvs4.py`, `craig.py`, `utils.py` used to produce the paper's
results), so the measured quantities are faithful to the method:

  * candidate pool H_c  = CHVS4_Algorithm3_Ding2017(Z_c, budget = alpha*r_c)   [chvs4.py]
  * coreset      S_c    = craig_greedy_cdist(Z_c, r_c, candidates = H_c)        [craig.py]
  * count weights gamma = calculate_weights(Z_c, S_c)                           [utils.py]
  * logit gradients z_i = softmax(logits) - onehot(y_i)                         [utils.py]

The per-class budget r_c and its remainder distribution replicate
`craigch.select_craig_ch_coreset` exactly.

Reported theory quantities (Euclidean distance d(z_i,z_j)=||z_i-z_j||_2,
as in the manuscript; note the selection code itself minimises d^2, which
yields the SAME Voronoi assignment and weights):

  * eps_hat_c    (Prop. certificate)  = max_{i in I_c} min_{h in H_c} d(z_i,z_h)
  * coverage loss L_c(S_c)            = sum_{i in I_c} min_{j in S_c} d(z_i,z_j)
  * covering radius R_c(S_c)          = max_{i in I_c} min_{j in S_c} d(z_i,z_j)
  * relative gradient-matching error  = || sum z_i - sum gamma_j z_j || / || sum z_i ||
    (aggregated over classes, as in coreset_selector.compute_grad_dist_for_current_coreset)
"""
from __future__ import annotations
import os
import sys
import time
import zipfile
import numpy as np


# --------------------------------------------------------------------------- #
#  Locate and import the real pipeline (CRAIG-CH-main)                         #
# --------------------------------------------------------------------------- #

def ensure_pipeline_on_path(code_dir: str | None = None) -> str:
    """Make the main pipeline (chv_craig / CRAIG-CH-main) importable and
    return its path.

    Search order: explicit code_dir, then `../chv_craig` (repository layout),
    then `../CRAIG-CH-main` (legacy layout). If only the zip
    `CRAIG-CH-main.zip` is present, extract it next to it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if code_dir:
        candidates.append(code_dir)
    candidates += [
        os.path.join(os.path.dirname(here), "chv_craig"),
        os.path.join(here, "CRAIG-CH-main"),
        os.path.join(os.path.dirname(here), "CRAIG-CH-main"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "chvs4.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    # try extracting the zip found in parent (Experiment_Results_New)
    parent = os.path.dirname(here)
    zpath = os.path.join(parent, "CRAIG-CH-main.zip")
    if os.path.isfile(zpath):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(parent)
        c = os.path.join(parent, "CRAIG-CH-main")
        if os.path.isfile(os.path.join(c, "chvs4.py")):
            sys.path.insert(0, c)
            return c
    raise FileNotFoundError(
        "Could not find CRAIG-CH-main (with chvs4.py). Pass --code_dir "
        "pointing to the extracted pipeline folder.")


# --------------------------------------------------------------------------- #
#  Datasets and models (mirrors the train scripts)                            #
# --------------------------------------------------------------------------- #

def build_dataset_and_model(dataset: str, device: str, data_root: str = "./data",
                            selection_no_aug: bool = True):
    """Return (trainset_for_selection, trainset_for_training, testset, model,
    num_classes). Mirrors the train scripts: model = LogReg(784,10) for
    FashionMNIST, resnet20(10) for CIFAR-10 / SVHN.

    `selection_no_aug=True` builds a NON-augmented copy for gradient /
    selection computations (deterministic gradients), which is the sensible
    choice for coreset selection; set False to exactly reproduce the training
    scripts, which select on the augmented `trainset_full`.
    """
    import torch
    import torchvision
    import torchvision.transforms as T
    from utils import load_fashion_mnist_all, load_cifar10_all, load_svhn_all
    from model_logistic_regression import LogisticRegression
    from model_resnet import resnet20

    dataset = dataset.lower()
    if dataset == "fashionmnist":
        train_full, testset = load_fashion_mnist_all(os.path.join(data_root, "fmnist"))
        sel = train_full  # logreg: no augmentation anyway
        model = LogisticRegression(input_dim=784, num_classes=10).to(device)
        num_classes = 10
    elif dataset in ("cifar10", "cifar"):
        train_full, testset = load_cifar10_all(os.path.join(data_root, "cifar10"))
        model = resnet20(num_classes=10).to(device)
        num_classes = 10
        if selection_no_aug:
            tfm = T.Compose([T.ToTensor(),
                             T.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))])
            sel = torchvision.datasets.CIFAR10(os.path.join(data_root, "cifar10"),
                                               train=True, download=True, transform=tfm)
        else:
            sel = train_full
    elif dataset == "svhn":
        train_full, testset = load_svhn_all(os.path.join(data_root, "svhn"))
        model = resnet20(num_classes=10).to(device)
        num_classes = 10
        if selection_no_aug:
            tfm = T.Compose([T.ToTensor(),
                             T.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970))])
            sel = torchvision.datasets.SVHN(os.path.join(data_root, "svhn"),
                                            split="train", download=True, transform=tfm)
        else:
            sel = train_full
    else:
        raise ValueError(f"unknown dataset {dataset}")
    return sel, train_full, testset, model, num_classes


def warmup_model(model, trainset, testset, epochs, device, lr=None, batch_size=128):
    """Short full-data warm-up so that logit gradients are informative, matching
    the manuscript protocol (10 epochs for FashionMNIST/SVHN, 20 for CIFAR-10).
    Returns the trained model (in place)."""
    import torch
    import torch.nn as nn
    from utils import evaluate
    is_logreg = model.__class__.__name__ == "LogisticRegression"
    if lr is None:
        lr = 0.05 if is_logreg else 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=0.0 if is_logreg else 0.9,
                          weight_decay=0.0 if is_logreg else 1e-4)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
    model.train()
    for ep in range(epochs):
        cur_lr = lr * (ep + 1) / max(1, epochs)      # linear warm-up, as in scripts
        for g in opt.param_groups:
            g["lr"] = cur_lr
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    try:
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
        acc = evaluate(model, testloader, device)
        print(f"   [warmup] {epochs} epochs done, test acc = {acc:.2f}%")
    except Exception:
        pass
    return model


# --------------------------------------------------------------------------- #
#  Per-class budget allocation (identical to craigch.select_craig_ch_coreset)  #
# --------------------------------------------------------------------------- #

def allocate_budgets(indices_by_class, total_size):
    class_sizes = {c: len(idx) for c, idx in indices_by_class.items() if len(idx) > 0}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1
    return allocated


# --------------------------------------------------------------------------- #
#  Pure-NumPy metric helpers (Euclidean d = ||.||_2)                          #
# --------------------------------------------------------------------------- #

def _min_dist(Zq: np.ndarray, Zref: np.ndarray, block: int = 4096) -> np.ndarray:
    """For each row of Zq, min Euclidean distance to rows of Zref."""
    q2 = np.einsum("ij,ij->i", Zq, Zq)
    r2 = np.einsum("ij,ij->i", Zref, Zref)
    out = np.empty(Zq.shape[0], dtype=np.float64)
    for s in range(0, Zq.shape[0], block):
        e = min(s + block, Zq.shape[0])
        d2 = q2[s:e, None] - 2.0 * (Zq[s:e] @ Zref.T) + r2[None, :]
        np.maximum(d2, 0.0, out=d2)
        out[s:e] = np.sqrt(d2.min(axis=1))
    return out


def eps_hat(Zc: np.ndarray, H_local) -> float:
    """Certificate eps_hat_c = max_i min_{h in H_c} d(z_i, z_h)."""
    return float(_min_dist(Zc, Zc[np.asarray(H_local, dtype=int)]).max())


def coverage_and_radius(Zc: np.ndarray, S_local):
    """Return (coverage_loss = sum_i min_j d, covering_radius = max_i min_j d)."""
    d = _min_dist(Zc, Zc[np.asarray(S_local, dtype=int)])
    return float(d.sum()), float(d.max())


def weighted_class_sum(Zc: np.ndarray, weights_dict) -> np.ndarray:
    """sum_{j in S_c} gamma_j z_j  (count weights)."""
    out = np.zeros(Zc.shape[1], dtype=np.float64)
    for j, w in weights_dict.items():
        out += float(w) * Zc[int(j)]
    return out
