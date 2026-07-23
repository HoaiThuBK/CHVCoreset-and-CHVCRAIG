#!/usr/bin/env bash
#
# run_local.sh  --  run the Section 3.1 measurements (and optionally the
# Section 3.2 alpha ablation) on Linux/macOS (bash).
#
# Automatically: creates a virtualenv, installs dependencies, runs
# measure_theory.py (which locates the main pipeline folder ../src),
# then exports the LaTeX tables.
#
# USAGE (open a terminal, cd into the folder containing this file):
#     # Fastest: FashionMNIST only on CPU (no GPU needed, a few minutes)
#     ./run_local.sh
#
#     # Add CIFAR-10 / SVHN (GPU recommended; use --gpu for the CUDA torch build)
#     ./run_local.sh --datasets fashionmnist,cifar10,svhn --device cuda --gpu
#
#     # Also run the alpha ablation (Section 3.2) on FashionMNIST
#     ./run_local.sh --ablation
#
# If the script is not executable, run once:  chmod +x run_local.sh
#
set -euo pipefail

# ---------- Defaults (override via flags below) ----------
DATASETS="fashionmnist"          # comma-separated: fashionmnist,cifar10,svhn
FRACTIONS="0.01,0.03,0.05"       # comma-separated
SEEDS="42,43,44"                 # comma-separated
ALPHA="5"
DEVICE="cpu"                     # "cpu" or "cuda"
GPU=0                            # 1 = install the CUDA torch build (default: CPU)
ABLATION=0                       # 1 = also run the alpha ablation (Section 3.2)
OUTDIR="out_theory"
ROOT="$HOME/chvexp"              # venv + data live here (kept separate from source)

# ---------- Parse command-line arguments ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)  DATASETS="$2"; shift 2 ;;
        --fractions) FRACTIONS="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --alpha)     ALPHA="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        --gpu)       GPU=1; shift ;;
        --ablation)  ABLATION=1; shift ;;
        --outdir)    OUTDIR="$2"; shift 2 ;;
        --root)      ROOT="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Turn comma-separated lists into bash arrays (passed space-separated to python).
IFS=',' read -r -a DATASET_ARR   <<< "$DATASETS"
IFS=',' read -r -a FRACTION_ARR  <<< "$FRACTIONS"
IFS=',' read -r -a SEED_ARR      <<< "$SEEDS"

# ---------- Locate the script's own directory and cd into it ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo -e "\033[36m==> Working directory: $SCRIPT_DIR\033[0m"

# ---------- 1) Find Python 3.12/3.11/3.10 (PyTorch does not yet support 3.13/3.14) ----------
find_base_python() {
    local c ver
    for c in python3.12 python3.11 python3.10; do
        if command -v "$c" >/dev/null 2>&1; then echo "$c"; return 0; fi
    done
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then
            ver="$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
            if [[ "$ver" =~ ^3\.(9|10|11|12)$ ]]; then echo "$c"; return 0; fi
        fi
    done
    return 1
}
if ! BASE_PY="$(find_base_python)"; then
    echo "PyTorch does not yet support Python 3.13/3.14. Install Python 3.12 (e.g. via your" >&2
    echo "package manager or pyenv), make sure it is on PATH, then rerun this script." >&2
    exit 1
fi
echo -e "\033[36m==> Base Python: $BASE_PY ($($BASE_PY --version 2>&1))\033[0m"

# ---------- 2) Create the venv + data dir ----------
mkdir -p "$ROOT"
DATA_ROOT="$ROOT/data"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"

need_create=1
if [[ -x "$VPY" ]]; then
    vver="$("$VPY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || true)"
    if [[ "$vver" =~ ^3\.(9|10|11|12)$ ]]; then
        need_create=0
    else
        echo -e "\033[33m==> Old venv uses Python $vver -> deleting and recreating ...\033[0m"
        rm -rf "$VENV"
    fi
fi
if [[ "$need_create" -eq 1 ]]; then
    echo -e "\033[36m==> Creating virtualenv at $VENV ...\033[0m"
    "$BASE_PY" -m venv "$VENV"
fi

# ---------- 3) Install dependencies ----------
echo -e "\033[36m==> Installing dependencies (numpy, scikit-learn, torch, torchvision) ...\033[0m"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install numpy scikit-learn tqdm

# torch + torchvision versions must MATCH (install in one command so pip picks a compatible pair).
if [[ "$GPU" -eq 1 ]]; then
    IDX=()
else
    IDX=(--index-url https://download.pytorch.org/whl/cpu)
fi
"$VPY" -m pip install torch torchvision "${IDX[@]}"

# Verify; on failure (e.g. broken torch wheel) clean up and reinstall from scratch.
if ! "$VPY" -c "import torch, torchvision" 2>/dev/null; then
    echo -e "\033[33m==> Broken torch install -> cleaning up and reinstalling ...\033[0m"
    "$VPY" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    SP="$("$VPY" -c 'import site;print(site.getsitepackages()[0])')"
    for d in torch torchgen torchvision torchaudio functorch; do
        rm -rf "$SP/$d"
    done
    "$VPY" -m pip cache purge 2>/dev/null || true
    "$VPY" -m pip install --no-cache-dir --force-reinstall torch torchvision "${IDX[@]}"
    if ! "$VPY" -c "import torch, torchvision" 2>/dev/null; then
        echo -e "\033[31mStill failing. Report the following two outputs to pin the correct versions:\033[0m"
        "$VPY" --version
        "$VPY" -m pip show torch torchvision
        echo "torch/torchvision still cannot be imported." >&2
        exit 1
    fi
fi
"$VPY" -c "import torch, torchvision; print('   torch', torch.__version__, '| torchvision', torchvision.__version__)"

# ---------- 4) Warm-up per dataset ----------
WARM=()
for d in "${DATASET_ARR[@]}"; do
    case "$(echo "$d" | tr '[:upper:]' '[:lower:]')" in
        cifar10) WARM+=("cifar10:20") ;;
        svhn)    WARM+=("svhn:10") ;;
        *)       WARM+=("fashionmnist:10") ;;
    esac
done

# ---------- 5) Run the Section 3.1 measurements ----------
echo -e "\033[32m==> measure_theory.py  (datasets: ${DATASET_ARR[*]} | device: $DEVICE) ...\033[0m"
"$VPY" measure_theory.py --datasets "${DATASET_ARR[@]}" --fractions "${FRACTION_ARR[@]}" \
    --seeds "${SEED_ARR[@]}" --alpha "$ALPHA" --warmup "${WARM[@]}" \
    --device "$DEVICE" --data_root "$DATA_ROOT" --outdir "$OUTDIR"

# ---------- 6) Export LaTeX tables ----------
echo -e "\033[32m==> Exporting LaTeX tables (Section 3.1) ...\033[0m"
TEX_OUT="$OUTDIR/table_theory.tex"
"$VPY" make_latex_tables.py --theory "$OUTDIR/theory_summary_*.csv" | tee "$TEX_OUT"

# ---------- 7) (Optional) alpha ablation, Section 3.2 ----------
if [[ "$ABLATION" -eq 1 ]]; then
    echo -e "\033[32m==> ablation.py (alpha = 3,5,10) on FashionMNIST ...\033[0m"
    mkdir -p out_ablation
    "$VPY" ablation.py --dataset fashionmnist --fraction 0.03 --alphas 3 5 10 \
        --seeds "${SEED_ARR[@]}" --warmup 10 --device "$DEVICE" \
        --data_root "$DATA_ROOT" --outdir "out_ablation"
    "$VPY" make_latex_tables.py --alpha "out_ablation/ablation_alpha_fashionmnist.csv" \
        | tee "out_ablation/table_alpha.tex"
fi

echo -e "\n\033[32m==> DONE.\033[0m"
echo -e "\033[32m    Result CSVs : $OUTDIR/theory_summary_*.csv\033[0m"
echo -e "\033[32m    LaTeX table : $TEX_OUT\033[0m"