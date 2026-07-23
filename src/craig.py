"""
---file craig.py---
Select a coreset via CRAIG greedy facility-location
"""
import time
import torch
from tqdm import tqdm
from utils import compute_gradient_representations, calculate_weights

def craig_greedy_cdist(
    gradients,
    budget,
    candidate_indices_local=None,
    desc="craig",
    candidate_batch_size=512,
):
    """
    Select a coreset via CRAIG greedy facility-location: iterate `budget` times, each time picking the candidate that most reduces the minimum squared
    distances (min_dists) from every point to the selected set (submodular greedy), computed batch-by-batch with torch.cdist to save memory.

    :param gradients: Gradient tensor of all samples, of shape (n, feature_dim).
    :param budget: Number of samples to select into the coreset.
    :param candidate_indices_local: List of local indices (into `gradients`) that are allowed to be selected. Default None means all n samples are used as candidates.
    :param desc: Label shown on the progress bar (tqdm).
    :param candidate_batch_size: Candidate batch size when computing cdist, used to reduce memory usage.
    :return: List of local indices (into `gradients`) of the samples selected into the coreset. If budget >= the number of candidates, returns all candidates directly (trimmed to budget).

    """
    gradients = gradients.detach().cpu().float()
    n = gradients.shape[0]
    if candidate_indices_local is None:
        candidate_indices_local = list(range(n))
    candidate_indices_local = list(dict.fromkeys(candidate_indices_local))
    if budget >= len(candidate_indices_local):
        return candidate_indices_local[:budget]
    selected_local = []
    selected_set = set()
    gradients = gradients.float()
    min_dists = torch.full(
        (n,),
        float("inf"),
        dtype=gradients.dtype,
        device=gradients.device,
    )
    for _ in tqdm(range(budget), desc=desc):
        remaining = [
            idx for idx in candidate_indices_local
            if idx not in selected_set
        ]
        if len(remaining) == 0:
            break
        best_gain = -1.0
        best_idx = -1
        remaining_tensor = torch.tensor(remaining, dtype=torch.long, device=gradients.device)
        for start in range(0, len(remaining), candidate_batch_size):
            batch_idx = remaining_tensor[start:start + candidate_batch_size]
            candidate_grads = gradients[batch_idx]
            
            # dists shape: [n_samples, batch_candidates]
            dists = torch.cdist(gradients, candidate_grads, p=2,) ** 2

            # gain of adding each candidate
            gains = torch.clamp(min_dists.unsqueeze(1) - dists, min=0.0)
            gain_per_candidate = gains.sum(dim=0)
            batch_best_pos = torch.argmax(gain_per_candidate)
            batch_best_gain = gain_per_candidate[batch_best_pos].item()

            if batch_best_gain > best_gain:
                best_gain = batch_best_gain
                best_idx = int(batch_idx[batch_best_pos].item())
        if best_idx == -1:
            break
        selected_local.append(best_idx)
        selected_set.add(best_idx)
        dist_to_new_member = torch.cdist(gradients, gradients[best_idx].view(1, -1), p=2).squeeze(1) ** 2
        min_dists = torch.minimum(min_dists, dist_to_new_member)
    return selected_local


def select_craig_coreset(
    # Mode 1: CRAIG for full dataset
    model,
    full_dataset,
    indices_by_class,
    coreset_size_fraction,
    device,
    gradient_type,
    num_classes=10,
    batch_size=128,

    # Mode 2: CRAIG for CHV-CRAIG
    gradients=None,
    budget=None,
    candidate_indices_local=None,
    desc="CRAIG",

    # cdist optimization
    candidate_batch_size=512,
):
    """
    Orchestrate coreset selection with CRAIG, supporting two modes:
    (1) run over the entire `full_dataset` per class, computing the gradients and the per-class budget automatically;
    (2) run directly on a precomputed `gradients` / `candidate_indices_local` matrix (used by CRAIG-CH), delegating to craig_greedy_cdist.

    :param model: Model used to compute the gradients (mode 1).
    :param full_dataset: The complete original dataset (mode 1).
    :param indices_by_class: Dict mapping a class label -> list of indices into `full_dataset` (mode 1).
    :param coreset_size_fraction: Coreset size as a fraction of `full_dataset` (mode 1).
    :param device: Device (CPU/GPU) used when computing gradients (mode 1).
    :param gradient_type: Type of gradient representation to compute (mode 1).
    :param num_classes: Total number of classes in the dataset (mode 1).
    :param batch_size: Batch size used when computing gradients (mode 1).
    :param gradients: Precomputed gradient matrix; if not None, the function switches to mode 2.
    :param budget: Number of samples to select (mode 2).
    :param candidate_indices_local: List of local candidate indices into `gradients` (mode 2).
    :param desc: Label shown on the progress bar (mode 2).
    :param candidate_batch_size: Candidate batch size when computing cdist.
    :return:
        - Mode 2: list of the selected local indices (returned directly from craig_greedy_cdist).
        - Mode 1: a tuple ``(final_indices, final_weights, elapsed_time)`` — the global indices of the coreset, their corresponding weights, and the elapsed time (in seconds).
    """
    # ==========================================================
    # Mode 2: CRAIG on gradient matrix/candidate (used for CHV-CRAIG)
    # ==========================================================
    if gradients is not None:
        return craig_greedy_cdist(
            gradients=gradients,
            budget=budget,
            candidate_indices_local=candidate_indices_local,
            desc=desc,
            candidate_batch_size=candidate_batch_size)

    # ==========================================================
    # Mode 1: CRAIG for full dataset, per-class
    # ==========================================================
    print(">>> Selecting coreset by CRAIG...")
    start_time = time.time()
    final_indices, final_weights = [], {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))
    class_sizes = {
        c: len(indices_by_class[c])
        for c in indices_by_class
        if len(indices_by_class[c]) > 0
    }
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    for c, class_budget in allocated.items():
        class_indices = indices_by_class[c]
        if class_budget <= 0 or len(class_indices) == 0:
            continue
        print(
            f"\n>>> CRAIG for class {c}: "
            f"budget={class_budget}, samples={len(class_indices)}"
        )

        class_gradients = compute_gradient_representations(
            model=model,
            full_dataset=full_dataset,
            indices=class_indices,
            device=device,
            gradient_type=gradient_type,
            batch_size=batch_size,
            return_device="device")

        if class_gradients.shape[0] == 0:
            continue
        selected_local = craig_greedy_cdist(
            gradients=class_gradients,
            budget=class_budget,
            candidate_indices_local=None,
            desc=f"CRAIG for class {c}",
            candidate_batch_size=candidate_batch_size)

        weights_local = calculate_weights(
            class_gradients=class_gradients,
            selected_local=selected_local)

        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()
    assert len(final_indices) == total_size, (
        f"CRAIG size mismatch: expected {total_size}, got {len(final_indices)}")
    print(f"\n>>> CRAIG selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")
    return final_indices, final_weights, end_time - start_time