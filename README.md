# CHVCoreset and CHV-CRAIG

This repository contains the official implementation for the paper:

> **Vu Hoai Thu, Nguyen Kieu Linh, Pham Thanh Hieu, CHVCoreset and CHV-CRAIG: Convex Hull Filtering for Scalable Gradient-Based Coreset Selection**
---

## 1. What each file does

**Coreset-selection algorithms.**
- `chvcoreset.py` — CHVS4 (Convex Hull Vertex Selection v4): approximates the
  convex hull of the per-class gradient representations. Builds an initial
  d-simplex (Wang 2013, Algorithm 1) and expands it facet-by-facet toward the
  farthest outlying point (Ding 2017, Algorithm 2–3). Source paper:
  https://tinyurl.com/2h7madsu.

- `craig.py` — CRAIG greedy facility-location selection: repeatedly picks the
  candidate that most reduces the maximum distance from every point to its
  nearest selected point, computed in batches with `torch.cdist`. Exposes
  `craig_greedy_cdist` (works on any gradient matrix) and
  `select_craig_coreset` (per-class driver for the full dataset).

- `craigch.py` — CRAIG-CH: hybrid selection that narrows the candidate pool
  with convex-hull vertices before/alongside running CRAIG's greedy selection
  (driven by the `candidate_multiplier` parameter). *(Not included among the
  files reviewed for docstrings in this conversation — described here only
  from its usage in `coreset_selector.py`.)*

- `random_selector.py` — uniform random per-class coreset baseline.
  *(Not included among the files reviewed for docstrings in this
  conversation.)*

**Shared infrastructure.**
- `coreset_selector.py` — `CoresetSelector`, a dispatcher class that wraps all
  four selection methods behind a single `select(method=...)` call, plus
  `compute_grad_dist_for_current_coreset` (gradient-matching diagnostic) and
  `WeightedSubsetDataset` (a `Dataset` that yields `(x, y, weight)` for
  weighted training on a selected coreset).

- `utils.py` — dataset loaders (`load_fashion_mnist_all`, `load_cifar10_all`,
  `load_svhn_all`), `get_indices_by_class`, per-sample gradient representation
  (`compute_gradient_representations`, `logit` or `embedding` mode),
  `train_one_epoch` (supports per-sample weighted loss), `evaluate`, and
  `calculate_weights` (nearest-selected-point weighting used by
  CRAIG/CHVS4/CRAIG-CH).

- `model_resnet.py` — ResNet-20 (CIFAR-style: 3 stages × 3 `BasicBlock`s,
  Kaiming init), with an `embedding()` method that exposes pre-logit features
  for gradient-representation computation.

- `model_logistic_regression.py` — Logistic Regression model used for the
  FashionMNIST experiments. 

**Training drivers.**
- `train_logreg_fmnist.py` — Logistic Regression on FashionMNIST (10 classes).
- `train_resnet20_cifar10.py` — ResNet-20 on CIFAR-10 (10 classes).
- `train_resnet20_svhn.py` — ResNet-20 on SVHN (10 classes).

Each driver follows the same loop: warm up on the full dataset for
`--warmup_epochs` epochs, then every `--update_freq` epochs re-select the
coreset with the chosen method and train on it (weighted loss) until the next
update. Per-epoch metrics (`loss`, `accuracy`, `lr`, `train_time_s`,
`selection_time_s`, `coreset_size`) are appended to a CSV file as training
proceeds.

---

## 2. Requirements

- Python 3.9+
- Packages: `numpy`, `torch`, `torchvision`, `scikit-learn`, `tqdm`

```bash
# Linux / macOS
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Windows
py -m venv venv && venv\Scripts\activate
py -m pip install -r requirements.txt
```

---

## 3. Folder layout

```
.
├── chvs4.py                    # CHVS4 convex-hull vertex selection
├── craig.py                    # CRAIG greedy facility-location selection
├── craigch.py                  # CRAIG-CH hybrid selection
├── random_selector.py          # Random baseline selection
├── coreset_selector.py         # CoresetSelector dispatcher, WeightedSubsetDataset
├── utils.py                    # Data loading, gradient repr., train/eval loops
├── model_resnet.py             # ResNet-20 definition
├── model_logistic_regression.py# Logistic Regression definition
├── train_logreg_fmnist.py      # Driver: Logistic Regression on FashionMNIST
├── train_resnet20_cifar10.py   # Driver: ResNet-20 on CIFAR-10
├── train_resnet20_svhn.py      # Driver: ResNet-20 on SVHN
├── requirements.txt
└── README.md                   # this file
```

`train_logreg_mnist.py`, `train_resnet20_cifar10.py`, and `train_resnet20_svhn.py` write their CSV results
under `results_mnist/<method>/`, `results_cifar10/<method>/` and `results_svhn/<method>/` respectively
(created automatically). 

---

## 4. Quick start

```bash
# Logistic Regression on FashionMNIST, CRAIG coreset at 10% of the data
python train_logreg_fmnist.py --selection_method craig --coreset_fraction 0.1

# ResNet-20 on CIFAR-10, CHVS4 coreset at 5% of the data
python train_resnet20_cifar10.py --selection_method chvs4 --coreset_fraction 0.05

# ResNet-20 on SVHN, CRAIG-CH coreset at 1% of the data
python train_resnet20_svhn.py --selection_method craig_ch --coreset_fraction 0.01

# Full-dataset baseline (no coreset selection) for any driver
python train_logreg_fmnist.py --selection_method full_dataset
```

---

## 5. Methods compared

| Name         | Source file            | Description                                                        |
|--------------|-------------------------|----------------------------------------------------------------------|
| Full dataset | training drivers (`full_dataset`) | Baseline: train on the entire dataset every epoch          |
| CRAIG        | `craig.py`               | Greedy facility-location coreset selection                       |
| CHVS4        | `chvs4.py`                | Convex-hull vertex approximation per class (Wang 2013 + Ding 2017) |
| CRAIG-CH     | `craigch.py`              | Hybrid: convex-hull candidate narrowing + CRAIG greedy selection  |
| Random       | `random_selector.py`      | Uniform random per-class sampling baseline                        |

---

## 6. Datasets

| Dataset      | Loader (in `utils.py`)  | Classes | Notes                                              |
|--------------|--------------------------|--------:|-----------------------------------------------------|
| FashionMNIST | `load_fashion_mnist_all` |      10 | Flattened 784-dim vectors; paired with Logistic Regression |
| CIFAR-10     | `load_cifar10_all`       |      10 | RandomCrop + HorizontalFlip augmentation on train; paired with ResNet-20 |
| SVHN         | `load_svhn_all`          |      10 | RandomCrop augmentation on train; paired with ResNet-20  |

---

## 7. What each run produces

Each training driver writes one CSV with the columns:

```
epoch, loss, accuracy, lr, train_time_s, selection_time_s, coreset_size
```

| Driver                        | Output path                                                                 |
|--------------------------------|-------------------------------------------------------------------------------|
| `train_logreg_fmnist.py`       | `results_mnist/<method>/results_mnist_logreg_<method>_<gradient_type>_seed<seed>[_frac<f>].csv` |
| `train_resnet20_cifar10.py`    | `results_cifar10_resnet20_<method>_<gradient_type>[_frac<f>].csv` (current directory) |
| `train_resnet20_svhn.py`       | `results_svhn/<method>/results_svhn_resnet20_<method>_<gradient_type>_seed<seed>[_frac<f>].csv` |

---

## 8. Notes on the training loop

- **Gradient representation.** `--gradient_type logit` uses `softmax(logits) -
  one_hot(label)` as the per-sample gradient proxy; `--gradient_type
  embedding` additionally outer-products that term with the model's
  pre-logit embedding.
- **Learning-rate schedule.** Handled manually in each driver's
  `get_current_lr()` — linear warm-up over `--warmup_epochs`, then a
  `--coreset_lr_scale` cut once training switches to a coreset, with further
  step decays of `--lr_gamma` at 50% and 75% of `--epochs`. No `torch` LR
  scheduler object is used.
- **Weighting.** CRAIG, CHVS4, and CRAIG-CH assign each selected sample a
  weight equal to the number of dataset points for which it is the nearest
  selected point (`calculate_weights` in `utils.py`); this weight is used in
  the per-sample-weighted loss during coreset training.
- **Per-class budgeting.** All selection methods allocate the total coreset
  budget across classes proportionally to class size, with leftover slots
  assigned to the classes with the largest fractional remainder.

---

## 9. Contact

For questions about the code or to report a reproduction issue, please contact
`thuvh@ptit.edu.vn`.