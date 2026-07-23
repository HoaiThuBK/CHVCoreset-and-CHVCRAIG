"""
---file random_selector.py---
Select a coreset via random 
"""

import random
import time
from utils import compute_gradient_representations, calculate_weights

def select_random_coreset(
    model,
    full_dataset,
    indices_by_class,
    coreset_size_fraction,
    device,
    gradient_type,
    num_classes,
    random_state,
    batch_size
    
):
    print(">>> Selecting coreset by Random...")
    start_time = time.time()
    rng = random.Random(random_state)
    final_indices = []
    final_weights = {}
    total_size = max(1, int(len(full_dataset) * coreset_size_fraction))
    class_sizes = {
        c: len(indices_by_class[c])
        for c in indices_by_class
        if len(indices_by_class[c]) > 0
    }
    total_n = sum(class_sizes.values())
    allocated = {c: int(total_size * class_sizes[c] / total_n)for c in class_sizes}
    remaining = total_size - sum(allocated.values())
    remainders = sorted(
        class_sizes.keys(),
        key=lambda c: -((total_size * class_sizes[c] / total_n) - allocated[c])
    )
    for c in remainders:
        if remaining <= 0:
            break
        allocated[c] += 1
        remaining -= 1
    for c, budget in allocated.items():
        class_indices = indices_by_class[c]
        if budget <= 0:
            continue
        class_gradients = compute_gradient_representations(
            model=model,
            full_dataset=full_dataset,
            indices=class_indices,
            device=device,
            gradient_type=gradient_type,
            batch_size=batch_size,
        )
        n_c = len(class_indices)
        if budget >= n_c:
            selected_local = list(range(n_c))
        else:
            selected_local = rng.sample(range(n_c), budget)
        weights_local = calculate_weights(class_gradients=class_gradients, selected_local=selected_local,
        )
        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)
    end_time = time.time()
    print(f">>> Random selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")
    return final_indices, final_weights, end_time - start_time