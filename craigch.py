"""
---file craig.py---
Chọn coreset bằng thuật toán kết hợp CRAIG-CH
"""
import time
from chvs4 import CHVS4_Algorithm3_Ding2017
from utils import compute_gradient_representations, calculate_weights
from craig import select_craig_coreset


def select_craig_ch_coreset(
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
    Hàm điều phối việc chọn coreset bằng CRAIG-CH theo từng lớp (class): Chọn các candidates budget bằng CHVS4, sau đó Greedy trên candidates bằng CRAIG để thu được budget coreset
    :param model: Mô hình dùng để tính gradient cho từng mẫu.
    :param full_dataset: Toàn bộ tập dữ liệu gốc.
    :param indices_by_class: Dict ánh xạ nhãn lớp -> danh sách chỉ số (trong full_dataset).
    :param coreset_size_fraction: Tỷ lệ kích thước coreset so với full_dataset.
    :param device: Thiết bị (CPU/GPU) dùng khi tính gradient.
    :param gradient_type: Loại biểu diễn gradient cần tính.
    :param num_classes: Tổng số lớp trong tập dữ liệu.
    :praram candidate_multiplier: Quy định số điểm candidates được chọn bằng CHVS4. 
    :param random_state: Seed cho các bước ngẫu nhiên trong CHVS4.
    :param batch_size: Kích thước batch khi tính gradient.
    :param chvs4_epsilon: Ngưỡng dừng epsilon truyền vào CHVS4_Algorithm3_Ding2017.
    :return: Bộ ba (final_indices, final_weights, elapsed_time) — chỉ số toàn cục của coreset, trọng số tương ứng, và thời gian thực hiện (giây).
    """
    print(">>> Selecting coreset by CRAIG-CH...")
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
            f"Class {c}: CHVS4 found {len(candidate_indices_local)} candidates "
            f"(unique={num_unique_candidates}) / "
            f"candidate_budget={candidate_budget}, final_budget={budget}, "
            f"class_samples={n_c}")

        if num_unique_candidates < len(candidate_indices_local):
            print(f"Warning: duplicated CHVS4 candidates in class {c}")
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
                desc=f"CRAIG-CH for class {c}",
                candidate_batch_size=512)

        weights_local = calculate_weights(class_gradients=gradients, selected_local=selected_local)
        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)
        end_time = time.time()
        print(f">>> CRAIG-CH selection done in {end_time - start_time:.2f}s")
        print(f">>> Coreset size: {len(final_indices)}")
    return final_indices, final_weights, end_time - start_time