"""
---file chv_craig.py---
Orchestrate per-class coreset selection using CHV-CRAIG: first select `candidate` points with CHVCoreset, then run Greedy CRAIG over those
candidates to obtain the final `budget` coreset.
"""
import time
from chvcoreset import CHVS4_Algorithm3_Ding2017
from utils import compute_gradient_representations, calculate_weights
from craig import select_craig_coreset


def select_chv_craig_coreset(
        model, 
        full_dataset, 
        indices_by_class, 
        coreset_size_fraction, 
        device, 
        gradient_type, 
        num_classes, 
        candidate_multiplier, 
        random_state, 
        batch_size, 
        chvs4_epsilon=0.0):
    """
        :param model: Model used to compute the per-sample gradients.
        :param full_dataset: The complete original dataset.
        :param indices_by_class: Dict mapping a class label -> list of indices (into `full_dataset`).
        :param coreset_size_fraction: Coreset size as a fraction of `full_dataset`.
        :param device: Device (CPU/GPU) used when computing gradients.
        :param gradient_type: Type of gradient representation to compute.
        :param num_classes: Total number of classes in the dataset.
        :param candidate_multiplier: Controls how many candidate points are selected by CHVCoreset.
        :param random_state: Seed for the randomized steps in CHVCoreset.
        :param batch_size: Batch size used when computing gradients.
        :param chvs4_epsilon: Epsilon stopping threshold passed to CHVS4_Algorithm3_Ding2017.
        :return: A tuple `(final_indices, final_weights, elapsed_time)` — the global indices of the coreset, their corresponding weights, and the elapsed time (in seconds).
    """
    print(">>> Selecting coreset by CHV-CRAIG...")
    start_time = time.time()
    final_indices, final_weights = [], {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))

    class_sizes = {c: len(indices_by_class[c]) for c in indices_by_class if len(indices_by_class[c]) > 0}
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n) for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    for c in sorted(allocated, key=lambda x: -((total_size * class_sizes[x] / total_n) - allocated[x])):
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1

    for c, budget in allocated.items():
        class_indices = indices_by_class[c]
        if budget <= 0 or len(class_indices) == 0:
            continue
        gradients = compute_gradient_representations(
            model=model,
            full_dataset=full_dataset,
            indices=class_indices,
            device=device,
            gradient_type=gradient_type,
            batch_size=batch_size)

        n_c = gradients.shape[0]
        candidate_budget = min(n_c, max(budget, candidate_multiplier * budget))

        candidate_indices_local = CHVS4_Algorithm3_Ding2017(
            gradients.detach().cpu().numpy(),
            epsilon=chvs4_epsilon,
            budget=candidate_budget,
            random_state=random_state
        ).astype(int).tolist()
        
        num_unique_candidates = len(set(candidate_indices_local))

        print(
            f"Class {c}: CHVCoreset found {len(candidate_indices_local)} candidates "
            f"(unique={num_unique_candidates}) / "
            f"candidate_budget={candidate_budget}, final_budget={budget}, "
            f"class_samples={n_c}")

        if num_unique_candidates < len(candidate_indices_local):
            print(f"Warning: duplicated CHVCoreset candidates in class {c}")
        if len(candidate_indices_local) == 0:
            continue
        if len(candidate_indices_local) <= budget:
            selected_local = candidate_indices_local
        else:
            selected_local = select_craig_coreset(
                model=model,
                full_dataset=full_dataset,
                indices_by_class=class_indices,
                coreset_size_fraction=coreset_size_fraction,
                device=device,
                gradient_type=gradient_type,
                num_classes=num_classes,
                batch_size=batch_size,
                gradients=gradients,
                budget=budget,
                candidate_indices_local=candidate_indices_local,
                desc=f"CHV-CRAIG for class {c}",
                candidate_batch_size=512)

        weights_local = calculate_weights(class_gradients=gradients, selected_local=selected_local)
        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)
        end_time = time.time()
        print(f">>> CHV-CRAIG selection done in {end_time - start_time:.2f}s")
        print(f">>> Coreset size: {len(final_indices)}")
    return final_indices, final_weights, end_time - start_time