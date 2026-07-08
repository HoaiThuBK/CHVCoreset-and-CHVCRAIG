"""
---file chvs4.py---
Code thuật toán CHVS4 (Convex Hull Coreset Selection version 4) tìm các đỉnh của bao lồi
Source paper: https://tinyurl.com/2h7madsu
"""
import numpy as np
import time
import random
import warnings
from numpy.linalg import norm, pinv
from utils import compute_gradient_representations
from craig import calculate_weights
from sklearn.decomposition import PCA

def dist_point_to_linear_subspace(p_vec: np.ndarray, basis_vectors_matrix: np.ndarray) -> float:
    """
    Hàm tính khoảng cách từ một điểm đến không gian con tuyến tính sinh bởi các vector cơ sở.
    (Dựa trên Wang 2013, Equation 2. Cần cho select_initial_simplex_Wang2013)
    :param p_vec: Vector điểm cần tính khoảng cách (u trong bài báo), đã tịnh tiến về gốc x0.
    :param basis_vectors_matrix: Ma trận vector cơ sở (U trong bài báo), mỗi cột một vector.
    :return: Khoảng cách Euclid từ p_vec đến không gian con. Trả về chuẩn của p_vec nếu
        basis_vectors_matrix rỗng hoặc gặp lỗi nghịch đảo.
    """
    t = basis_vectors_matrix.shape[1]
    if t == 0: return norm(p_vec)
    Q = basis_vectors_matrix.T @ basis_vectors_matrix
    c = basis_vectors_matrix.T @ p_vec
    try:
        a_star = pinv(Q) @ c
        dist_sq = p_vec @ p_vec - c @ a_star
        return np.sqrt(max(0.0, dist_sq))
    except np.linalg.LinAlgError:
        return norm(p_vec)

def select_initial_simplex_Wang2013(P: np.ndarray, budget: int = -1, random_state: int | None = None) -> list[int]:
    """
    Hàm: Chọn d+1 đỉnh khởi tạo cho một d-simplex xấp xỉ bao lồi của tập điểm P,
    theo Algorithm 1 của Wang (2013), bằng cách lặp lại việc tìm điểm xa nhất so với
    không gian con hiện có cho đến khi đủ đỉnh hoặc hết ngân sách.
    :param P: Ma trận tất cả các điểm trong tập dữ liệu, kích thước (n, d).
    :param budget: Số điểm tối đa được chọn. Mặc định -1 là không giới hạn.
    :param random_state: Seed cho bộ sinh số ngẫu nhiên.
    :return: Danh sách chỉ số (trong P) của các đỉnh simplex khởi tạo.
    """
    rng = random.Random(random_state)
    n, d = P.shape
    if n <= d:
        warnings.warn(f"Số điểm ({n}) không đủ để tạo simplex. Trả về tất cả.")
        return list(range(n))
    if budget != -1 and n > budget and d > budget:
         warnings.warn(f"Budget ({budget}) nhỏ hơn số chiều ({d}). Trả về {budget} điểm ngẫu nhiên.")
         return rng.sample(range(n), budget)
    rand_idx = rng.randint(0, n - 1)
    x0 = P[rand_idx, :]
    dists_x0 = norm(P - x0, axis=1)
    xj0_idx = np.argmax(dists_x0)
    xj0 = P[xj0_idx, :]
    dists_xj0 = norm(P - xj0, axis=1)
    xj1_idx = np.argmax(dists_xj0)
    S_t_indices = {rand_idx, xj0_idx, xj1_idx} 
    tilde_U_all = P - x0
    if xj0_idx == xj1_idx or xj0_idx == rand_idx or xj1_idx == rand_idx:
         unique_indices = list(S_t_indices)
         if not unique_indices: 
            return [rand_idx]
         tilde_U_S_t_basis = tilde_U_all[unique_indices, :].T
         t = len(unique_indices)
    else:
        tilde_U_S_t_basis = np.hstack([
            tilde_U_all[xj0_idx, :].reshape(-1, 1),
            tilde_U_all[xj1_idx, :].reshape(-1, 1)
        ])
        if rand_idx != xj0_idx and rand_idx != xj1_idx:
             S_t_indices.add(rand_idx)
        t = 2 
    while t < d and (budget == -1 or len(S_t_indices) < budget):
        all_candidate_dists = np.full(n, -np.inf)
        candidate_indices = [i for i in range(n) if i not in S_t_indices]
        if not candidate_indices: break 
        for i in candidate_indices:
            u_i = tilde_U_all[i, :]
            all_candidate_dists[i] = dist_point_to_linear_subspace(u_i, tilde_U_S_t_basis)
        if np.max(all_candidate_dists) <= 1e-9: break 
        best_idx_subspace = np.argmax(all_candidate_dists)
        S_t_indices.add(best_idx_subspace)
        tilde_U_S_t_basis = tilde_U_all[list(S_t_indices), :].T
        t += 1
    final_indices = list(S_t_indices)
    if budget != -1 and len(final_indices) > budget:
        final_indices = final_indices[:budget]
    return final_indices

def partition_points_Wang2013(P: np.ndarray, p_indices_to_partition: list[int], simplex_indices: list[int]) -> list[list[int]]:
    """
    Hàm: Phân hoạch các điểm vào từng mặt (facet) của một d-simplex, dựa trên tọa độ tỷ cự
    (barycentric coordinates): điểm p được gán vào phần ứng với mặt đối diện đỉnh có tọa độ
    tỷ cự âm nhỏ nhất; điểm nằm trong/trên simplex thì không được gán vào phần nào.
    :param P: Ma trận tất cả các điểm trong tập dữ liệu, kích thước (n, d).
    :param p_indices_to_partition: Danh sách chỉ số các điểm cần phân hoạch.
    :param simplex_indices: Danh sách d+1 chỉ số đỉnh tạo thành simplex làm cơ sở phân hoạch.
    :return: Danh sách (d+1) phần, mỗi phần là danh sách chỉ số điểm thuộc mặt đối diện
        đỉnh tương ứng trong simplex_indices.
    """
    num_vertices = len(simplex_indices)
    d = P.shape[1]
    if num_vertices != d + 1: return [[] for _ in range(num_vertices)]
    V_simplex_matrix = P[simplex_indices, :]
    parts = [[] for _ in range(num_vertices)]
    A_bary = np.vstack((V_simplex_matrix.T, np.ones(d + 1)))
    try: A_bary_inv = pinv(A_bary)
    except np.linalg.LinAlgError: return parts
    for p_idx in p_indices_to_partition:
        p = P[p_idx, :]
        b_bary = np.append(p, 1)
        alpha = A_bary_inv @ b_bary
        if np.min(alpha) < -1e-9:
            min_idx = np.argmin(alpha)
            parts[min_idx].append(p_idx)
    return parts

def get_facets_indices(simplex_indices: list[int]) -> list[list[int]]:
    """
    Hàm lấy chỉ số đỉnh của tất cả các mặt (facet) của một d-simplex: mỗi mặt là simplex con
    gồm d đỉnh còn lại sau khi loại bỏ một đỉnh.
    :param simplex_indices: Các chỉ số của d+1 đỉnh tạo thành simplex.
    :return: Danh sách d+1 mặt, mỗi mặt là danh sách d chỉ số đỉnh.
    """
    facets = []
    for i in range(len(simplex_indices)):
        facets.append(simplex_indices[:i] + simplex_indices[i+1:])
    return facets

def compute_projection_distance_dp(x: np.ndarray, S_points_matrix: np.ndarray) -> float:
    """
    Hàm tính khoảng cách từ một điểm x đến siêu phẳng (hyperplane) sinh bởi d điểm đầu tiên của
    S_points_matrix, dùng làm xấp xỉ khoảng cách từ x đến mặt (facet) của bao lồi.
    (Dựa trên Algorithm 2 Computation of the Distance Between a Point and a Hyperplane của Ding 2017)
    :param x: Điểm cần tính khoảng cách.
    :param S_points_matrix: Ma trận các điểm định nghĩa mặt/siêu phẳng (S trong Algorithm 2),
        mỗi hàng là một điểm, kích thước (n_S, d).
    :return: Khoảng cách từ x đến siêu phẳng. Nếu S_points_matrix rỗng hoặc thiếu điểm để xác
        định siêu phẳng, trả về xấp xỉ bằng khoảng cách nhỏ nhất đến từng điểm; nếu suy biến,
        trả về np.inf.
    """
    n_S, d = S_points_matrix.shape
    if n_S == 0: return norm(x) 
    if n_S < d: 
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])
    
    V_hyperplane = S_points_matrix[:d, :]
    p1 = V_hyperplane[0, :]
    basis_vectors = V_hyperplane[1:, :] - p1
    try: _, _, vh = np.linalg.svd(basis_vectors)
    except np.linalg.LinAlgError: 
        return np.min([norm(x - S_points_matrix[i,:]) for i in range(n_S)])
    beta = vh[-1]
    norm_beta = norm(beta)
    if norm_beta < 1e-12: return np.inf
    return np.abs(np.dot(beta, x - p1)) / norm_beta

def CHVS4_Algorithm3_Ding2017(P: np.ndarray, epsilon: float = 0.0, budget: int = -1, random_state: int | None = None,) -> np.ndarray:
    """
    Hàm chính của thuật toán CHVS4 (Algorithm 3 của Ding 2017): xấp xỉ bao lồi của tập điểm P
    bằng cách khởi tạo một d-simplex rồi lặp lại việc thêm điểm xa mặt hiện tại nhất vào tập
    đỉnh, "phình" simplex ra ngoài cho đến khi đạt epsilon, hết ngân sách, hoặc hết điểm mở rộng.
    :param P: Ma trận tất cả các điểm trong tập dữ liệu, kích thước (n, d).
    :param epsilon: Ngưỡng dừng: dừng khi khoảng cách xa nhất còn lại <= epsilon.
    :param budget: Số đỉnh tối đa được chọn. Mặc định -1 là không giới hạn.
    :param random_state: Seed cho bộ sinh số ngẫu nhiên, dùng khi chọn simplex khởi tạo.
    :return: Mảng chỉ số (trong P) của các điểm được chọn làm đỉnh xấp xỉ bao lồi.
    """
    n, d = P.shape
    if n <= d + 1: return np.array(list(range(n)))

    initial_simplex_indices = select_initial_simplex_Wang2013(P, budget=budget, random_state=random_state)
    final_V_indices = set(initial_simplex_indices)
    
    if budget != -1 and len(final_V_indices) >= budget:
        return np.array(list(final_V_indices))

    candidate_indices = [i for i in range(n) if i not in final_V_indices]
    
    if len(initial_simplex_indices) != d + 1:
        warnings.warn(f"Simplex ban đầu chỉ có {len(initial_simplex_indices)} điểm (do budget/chiều). Thuật toán CHVS4 dừng sớm.")
        return np.array(list(final_V_indices))

    initial_facets = get_facets_indices(initial_simplex_indices)
    initial_parts = partition_points_Wang2013(P, candidate_indices, initial_simplex_indices)
    active_problems = []
    for i in range(len(initial_facets)):
        part, facet = initial_parts[i], initial_facets[i]
        if not part: continue
        facet_matrix = P[facet, :]
        dists_in_part = [compute_projection_distance_dp(P[p_idx, :], facet_matrix) for p_idx in part]
        if not dists_in_part: continue 
        max_dist_idx_local = np.argmax(dists_in_part)
        active_problems.append({'part_indices': part, 'facet_indices': facet, 'max_dist': dists_in_part[max_dist_idx_local], 'farthest_point_idx': part[max_dist_idx_local]})
    
    max_iter, current_iter = 4 * n, 0
    while active_problems and current_iter < max_iter and (budget == -1 or len(final_V_indices) < budget):
        current_iter += 1
        best_problem = max(active_problems, key=lambda p: p['max_dist'])
        if best_problem['max_dist'] <= epsilon: break
        active_problems.remove(best_problem)
        
        x_j0_star_idx = best_problem['farthest_point_idx']
        final_V_indices.add(x_j0_star_idx)
        
        if budget != -1 and len(final_V_indices) >= budget:
            break

        old_part = [p for p in best_problem['part_indices'] if p != x_j0_star_idx]
        if not old_part: continue
        new_simplex_indices = best_problem['facet_indices'] + [x_j0_star_idx]
        new_parts = partition_points_Wang2013(P, old_part, new_simplex_indices)
        new_facets = get_facets_indices(new_simplex_indices)
        for i in range(len(new_facets)):
            part, facet = new_parts[i], new_facets[i]
            if not part: continue
            facet_matrix = P[facet, :]
            dists_in_part = [compute_projection_distance_dp(P[p_idx, :], facet_matrix) for p_idx in part]
            if not dists_in_part: continue 
            max_dist_idx_local = np.argmax(dists_in_part)
            active_problems.append({'part_indices': part, 'facet_indices': facet, 'max_dist': dists_in_part[max_dist_idx_local], 'farthest_point_idx': part[max_dist_idx_local]})
    
    final_indices_list = list(final_V_indices)
    if budget != -1 and len(final_indices_list) > budget:
        final_indices_list = final_indices_list[:budget]

    return np.array(final_indices_list)


def select_chvs4_coreset(
        model, 
        full_dataset, 
        indices_by_class, 
        coreset_size_fraction, 
        device, gradient_type, 
        num_classes, 
        random_state, 
        batch_size, 
        chvs4_epsilon=0.0,
        log_selected_indices=True,
    ):
    """
    Hàm điều phối việc chọn coreset bằng CHVS4 theo từng lớp (class): tính ngân sách cho mỗi
    lớp theo tỷ lệ coreset_size_fraction, tính gradient của các mẫu trong lớp, rồi gọi
    CHVS4_Algorithm3_Ding2017 để chọn đỉnh bao lồi xấp xỉ làm đại diện; bù/cắt cho đúng ngân
    sách nếu thiếu/thừa, và tính trọng số (weight) bằng calculate_weights.
    :param model: Mô hình dùng để tính gradient cho từng mẫu.
    :param full_dataset: Toàn bộ tập dữ liệu gốc.
    :param indices_by_class: Dict ánh xạ nhãn lớp -> danh sách chỉ số (trong full_dataset).
    :param coreset_size_fraction: Tỷ lệ kích thước coreset so với full_dataset.
    :param device: Thiết bị (CPU/GPU) dùng khi tính gradient.
    :param gradient_type: Loại biểu diễn gradient cần tính.
    :param num_classes: Tổng số lớp trong tập dữ liệu.
    :param random_state: Seed cho các bước ngẫu nhiên trong CHVS4.
    :param batch_size: Kích thước batch khi tính gradient.
    :param chvs4_epsilon: Ngưỡng dừng epsilon truyền vào CHVS4_Algorithm3_Ding2017.
    :param log_selected_indices: Nếu True, in log chỉ số cục bộ/toàn cục được chọn.
    :return: Bộ ba (final_indices, final_weights, elapsed_time) — chỉ số toàn cục của coreset, trọng số tương ứng, và thời gian thực hiện (giây).
    """
    print(">>> Selecting coreset by CHVS4...")
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
        print(f"\n>>> CHVS4 for class {c}: budget={budget}, samples={len(class_indices)}")

        class_gradients = compute_gradient_representations(model=model, full_dataset=full_dataset, indices=class_indices, device=device, gradient_type=gradient_type, batch_size=batch_size)

        X = class_gradients.detach().cpu().numpy().astype(np.float64)
        n_c = X.shape[0]
        if budget >= n_c:
            # Ngân sách >= số mẫu của lớp: lấy toàn bộ mẫu, không cần chạy CHVS4.
            selected_local = list(range(n_c))
            print(f"Class {c}: budget >= samples, lấy toàn bộ {n_c} mẫu / budget={budget}")
        else:
            selected_local = CHVS4_Algorithm3_Ding2017(P=X, epsilon=chvs4_epsilon, budget=budget, random_state=random_state).astype(int).tolist()
            selected_local = list(dict.fromkeys(selected_local))
            print(
                f"Class {c}: CHVS4 selected {len(selected_local)} vertices "
                f"/ budget={budget}")
            if len(selected_local) < budget:
                selected_set = set(selected_local)
                remaining_candidates = [
                    i for i in range(n_c)
                    if i not in selected_set
                ]

                need = budget - len(selected_local)
                selected_local.extend(remaining_candidates[:need])

            if len(selected_local) > budget:
                selected_local = selected_local[:budget]

        selected_global = [class_indices[i] for i in selected_local]
        if log_selected_indices:
            print(f"\n[CHVS4 selected indices] class={c}")
            print(f"local_indices : {selected_local}")
            print(f"global_indices: {selected_global}")
        
        weights_local = calculate_weights(
            class_gradients=class_gradients,
            selected_local=selected_local,
        )

        for local_idx, weight in weights_local.items():
            global_idx = class_indices[local_idx]
            final_indices.append(global_idx)
            final_weights[global_idx] = float(weight)

    end_time = time.time()
    print(f"\n>>> CHVS4 selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if log_selected_indices:
        print("\n[CHVS4 final coreset global indices]")
        print(final_indices)

    return final_indices, final_weights, end_time - start_time