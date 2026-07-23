"""
---file chvcoreset.py---
Convex-Hull-based coreset selection (CHVCoreset). This module approximates the convex hull of a set of gradient representations
and uses its vertices as a coreset.
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
    Compute the distance from a point to the linear subspace spanned by a set of basis vectors.
    (Based on Wang 2013, Equation 2. Required by select_initial_simplex_Wang2013.)

    :param p_vec: The point vector whose distance is computed (``u`` in the paper), already translated to the origin ``x0``.
    :param basis_vectors_matrix: Matrix of basis vectors (``U`` in the paper), one vector per column.
    :return: The Euclidean distance from ``p_vec`` to the subspace. 
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
    Select the d+1 initial vertices of a d-simplex that approximates the convex hull of the point set P, following Algorithm 1 of Wang (2013), 
    by repeatedly finding the point farthest from the current subspace until enough vertices are collected or the budget is exhausted.

    :param P: Matrix of all points in the dataset, of shape (n, d).
    :param budget: Maximum number of points to select. Default -1 means no limit.
    :param random_state: Seed for the random number generator.
    :return: List of indices (into P) of the initial simplex vertices.
    """
    rng = random.Random(random_state)
    n, d = P.shape
    if n <= d:
        warnings.warn(f"The number of points ({n}) is not enough to create a simplex. Return all.")
        return list(range(n))
    if budget != -1 and n > budget and d > budget:
         warnings.warn(f"Budget({budget}) is less than the dimension({d}). Returns {budget} random points.")
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
    Partition points into the facets of a d-simplex based on their barycentric coordinates: a point p is assigned to the region corresponding to the facet
    opposite the vertex for which it has the smallest (most negative) barycentric coordinate; points lying inside/on the simplex are not assigned to any region.

    :param P: Matrix of all points in the dataset, of shape (n, d).
    :param p_indices_to_partition: List of indices of the points to partition.
    :param simplex_indices: List of d+1 vertex indices forming the simplex used as the basis for the partition.
    :return: List of (d+1) regions, where each region is a list of point indices belonging to the facet opposite the corresponding vertex in ``simplex_indices``.
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
    Get the vertex indices of all facets of a d-simplex: each facet is the sub-simplex formed by the remaining d vertices after removing one vertex.

    :param simplex_indices: The indices of the d+1 vertices forming the simplex.
    :return: List of d+1 facets, where each facet is a list of d vertex indices.
    """
    facets = []
    for i in range(len(simplex_indices)):
        facets.append(simplex_indices[:i] + simplex_indices[i+1:])
    return facets

def compute_projection_distance_dp(x: np.ndarray, S_points_matrix: np.ndarray) -> float:
    """
    Compute the distance from a point x to the hyperplane spanned by the first d points of S_points_matrix, used as an approximation of the distance from x to
    a facet of the convex hull. (Based on Algorithm 2, "Computation of the Distance Between a Point and a Hyperplane", of Ding 2017.)

    :param x: The point whose distance is computed.
    :param S_points_matrix: Matrix of points defining the facet/hyperplane (S in Algorithm 2), one point per row, of shape (n_S, d).
    :return: The distance from x to the hyperplane. If S_points_matrix is empty or lacks enough points to define the hyperplane, returns an approximation
        based on the smallest distance to each individual point; if the system is degenerate, returns np.inf.
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
    Main routine of the CHVS4 algorithm (Algorithm 3 of Ding 2017): approximate the convex hull of the point set P by initializing a d-simplex and then
    repeatedly adding the point farthest from the current facets to the vertex set, "inflating" the simplex outward until epsilon is reached, the budget is
    exhausted, or no expanding point remains.

    :param P: Matrix of all points in the dataset, of shape (n, d).
    :param epsilon: Stopping threshold: stop when the largest remaining distance is <= epsilon.
    :param budget: Maximum number of vertices to select. Default -1 means no limit.
    :param random_state: Seed for the random number generator, used when selecting the initial simplex.
    :return: Array of indices (into P) of the points selected as vertices of the approximate convex hull.
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


def select_chvcoreset(
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
    Orchestrate per-class coreset selection using CHVCoreset: compute the budget for each class according to `coreset_size_fraction`, compute the gradients of
    the samples in the class, then call CHVS4_Algorithm3_Ding2017 to select the approximate convex-hull vertices as representatives; pad/trim to match the
    exact budget if there are too few/too many, and compute the weights via `calculate_weights`.

    :param model: Model used to compute the per-sample gradients.
    :param full_dataset: The complete original dataset.
    :param indices_by_class: Dict mapping a class label -> list of indices (into `full_dataset`).
    :param coreset_size_fraction: Coreset size as a fraction of `full_dataset`.
    :param device: Device (CPU/GPU) used when computing gradients.
    :param gradient_type: Type of gradient representation to compute.
    :param num_classes: Total number of classes in the dataset.
    :param random_state: Seed for the randomized steps in CHVS4.
    :param batch_size: Batch size used when computing gradients.
    :param chvs4_epsilon: Epsilon stopping threshold passed to CHVS4_Algorithm3_Ding2017.
    :param log_selected_indices: If True, log the selected local/global indices.
    :return: A tuple ``(final_indices, final_weights, elapsed_time)`` — the global indices of the coreset, their corresponding weights, and the elapsed time (in seconds).
    """
    print(">>> Selecting coreset by CHVCoreset...")
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
        print(f"\n>>> CHVCoreset for class {c}: budget={budget}, samples={len(class_indices)}")

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
                f"Class {c}: CHVCoreset selected {len(selected_local)} vertices "
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
            print(f"\n[CHVCoreset selected indices] class={c}")
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
    print(f"\n>>> CHVCoreset selection done in {end_time - start_time:.2f}s")
    print(f">>> Coreset size: {len(final_indices)}")

    if log_selected_indices:
        print("\n[CHVCoreset final coreset global indices]")
        print(final_indices)

    return final_indices, final_weights, end_time - start_time