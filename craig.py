"""
---file craig.py---
Chọn coreset bằng thuật toán CRAIG
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
    Hàm chọn coreset bằng CRAIG greedy facility-location: lặp `budget` lần, mỗi lần chọn ứng
    viên giúp giảm khoảng cách bình phương tối thiểu (min_dists) từ mọi điểm tới tập đã chọn
    nhiều nhất (submodular greedy), tính theo batch bằng torch.cdist để tiết kiệm bộ nhớ.
    :param gradients: Tensor gradient của tất cả các mẫu, kích thước (n, feature_dim).
    :param budget: Số lượng mẫu cần chọn vào coreset.
    :param candidate_indices_local: Danh sách chỉ số cục bộ (trong gradients) được phép chọn.Mặc định None nghĩa là dùng toàn bộ n mẫu làm ứng viên.
    :param desc: Nhãn hiển thị trên thanh tiến trình (tqdm).
    :param candidate_batch_size: Kích thước batch ứng viên khi tính cdist, giúp giảm bộ nhớ.
    :return: Danh sách chỉ số cục bộ (trong gradients) của các mẫu được chọn vào coreset. Nếu budget >= số ứng viên, trả về luôn toàn bộ ứng viên (đã cắt theo budget).
    """
    # Nếu muốn CRAIG chạy trên CPU
    gradients = gradients.detach().cpu().float()
    print(f"[CRAIG] gradients.device = {gradients.device}")
    print(f"[CRAIG] gradients.shape  = {gradients.shape}")
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
    # Mode 1: CRAIG cho full dataset
    model,
    full_dataset,
    indices_by_class,
    coreset_size_fraction,
    device,
    gradient_type,
    num_classes=10,
    batch_size=128,

    # Mode 2: CRAIG dùng cho CRAIG-CH / CHVD-CRAIG
    gradients=None,
    budget=None,
    candidate_indices_local=None,
    desc="CRAIG",

    # cdist optimization
    candidate_batch_size=512,
):
    """
    Hàm điều phối chọn coreset bằng CRAIG, hỗ trợ hai chế độ:
    (1) chạy trên toàn bộ full_dataset theo từng lớp, tự tính gradient và ngân sách mỗi lớp;
    (2) chạy trực tiếp trên ma trận `gradients`/`candidate_indices_local` có sẵn (dùng cho CRAIG-CH), gọi lại craig_greedy_cdist.
    :param model: Mô hình dùng để tính gradient (chế độ 1).
    :param full_dataset: Toàn bộ tập dữ liệu gốc (chế độ 1).
    :param indices_by_class: Dict ánh xạ nhãn lớp -> danh sách chỉ số trong full_dataset (chế độ 1).
    :param coreset_size_fraction: Tỷ lệ kích thước coreset so với full_dataset (chế độ 1).
    :param device: Thiết bị (CPU/GPU) dùng khi tính gradient (chế độ 1).
    :param gradient_type: Loại biểu diễn gradient cần tính (chế độ 1).
    :param num_classes: Tổng số lớp trong tập dữ liệu (chế độ 1).
    :param batch_size: Kích thước batch khi tính gradient (chế độ 1).
    :param gradients: Ma trận gradient có sẵn; nếu khác None, hàm chuyển sang chế độ 2.
    :param budget: Số lượng mẫu cần chọn (chế độ 2).
    :param candidate_indices_local: Danh sách chỉ số ứng viên cục bộ trong gradients (chế độ 2).
    :param desc: Nhãn hiển thị trên thanh tiến trình (chế độ 2).
    :param candidate_batch_size: Kích thước batch ứng viên khi tính cdist.
    :return:
        - Chế độ 2: danh sách chỉ số cục bộ được chọn (kết quả trực tiếp từ craig_greedy_cdist).
        - Chế độ 1: bộ ba (final_indices, final_weights, elapsed_time) — chỉ số toàn cục của
          coreset, trọng số tương ứng, và thời gian thực hiện (giây).
    """
    # ==========================================================
    # Mode 2: CRAIG trên ma trận gradient/candidate có sẵn (sử dụng cho CRAIG-CH)
    # ==========================================================
    if gradients is not None:
        return craig_greedy_cdist(
            gradients=gradients,
            budget=budget,
            candidate_indices_local=candidate_indices_local,
            desc=desc,
            candidate_batch_size=candidate_batch_size)

    # ==========================================================
    # Mode 1: CRAIG cho full dataset, chạy theo từng lớp
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