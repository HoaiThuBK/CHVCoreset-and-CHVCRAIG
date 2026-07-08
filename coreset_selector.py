import torch
import numpy as np
import random
from torch.utils.data import Dataset
from utils import compute_gradient_representations
from random_selector import select_random_coreset
from craig import select_craig_coreset
from chvs4 import select_chvs4_coreset
from craigch import select_craig_ch_coreset

def set_seed(seed: int):
    """
    Hàm đặt seed cho các bộ sinh số ngẫu nhiên (random, numpy, torch, cudnn) để tái lập kết quả.
    :param seed: Giá trị seed dùng để khởi tạo các bộ sinh số ngẫu nhiên.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class CoresetSelector:
    """
    Lớp điều phối việc chọn coreset theo nhiều phương pháp khác nhau (craig, chvs4, random, craig_ch).
    """
    def __init__(
        self,
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
        chvs4_epsilon=0.0
    ):
        """
        Hàm khởi tạo, lưu lại các tham số dùng chung cho mọi phương pháp chọn coreset.
        :param model: Mô hình dùng để tính gradient cho từng mẫu.
        :param full_dataset: Toàn bộ tập dữ liệu gốc.
        :param indices_by_class: Dict ánh xạ nhãn lớp sang danh sách chỉ số trong full_dataset.
        :param coreset_size_fraction: Tỷ lệ kích thước coreset so với full_dataset.
        :param device: Thiết bị (CPU/GPU) dùng khi tính gradient.
        :param gradient_type: Loại biểu diễn gradient cần tính.
        :param num_classes: Tổng số lớp trong tập dữ liệu.
        :param candidate_multiplier: Hệ số nhân quy định số candidates, dùng cho phương pháp craig_ch.
        :param random_state: Seed cho các bước ngẫu nhiên trong quá trình chọn coreset.
        :param batch_size: Kích thước batch khi tính gradient.
        :param chvs4_epsilon: Ngưỡng dừng epsilon dùng cho phương pháp chvs4/craig_ch.
        """
        self.model = model
        self.full_dataset = full_dataset
        self.indices_by_class = indices_by_class
        self.coreset_size_fraction = coreset_size_fraction
        self.device = device
        self.gradient_type = gradient_type
        self.num_classes = num_classes
        self.candidate_multiplier = candidate_multiplier
        self.random_state = random_state
        self.batch_size = batch_size
        self.chvs4_epsilon = chvs4_epsilon

    def select(self, method="craig"):
        """
        Hàm chọn coreset bằng phương pháp được chỉ định.
        :param method: Tên phương pháp chọn coreset ("craig", "chvs4", "random", "craig_ch").
        :return: Kết quả trả về từ hàm select tương ứng với phương pháp đã chọn.
        """
        if method == "craig":
            return select_craig_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.batch_size,
            )
        
        elif method == "chvs4":
            return select_chvs4_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.random_state,
                self.batch_size,
                self.chvs4_epsilon,
            )

        elif method == "random":
            return select_random_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type,
                self.num_classes,
                self.random_state,
                self.batch_size,
            )
        elif method == "craig_ch":
            return select_craig_ch_coreset(
                self.model,
                self.full_dataset,
                self.indices_by_class,
                self.coreset_size_fraction,
                self.device,
                self.gradient_type, 
                self.num_classes,
                self.candidate_multiplier,
                self.random_state,
                self.batch_size,
                self.chvs4_epsilon,
            )
        else:
            raise ValueError(f"Phương pháp lựa chọn không hợp lệ: {method}")
        
    def compute_grad_dist_for_current_coreset(self, coreset_indices, coreset_weights):
        """
        Hàm tính khoảng cách gradient chuẩn hóa giữa tổng gradient có trọng số của coreset và tổng gradient của toàn bộ tập dữ liệu, dùng để đánh giá chất lượng coreset.
        :param coreset_indices: Danh sách chỉ số toàn cục của các mẫu trong coreset.
        :param coreset_weights: Dict ánh xạ chỉ số toàn cục sang trọng số của mẫu trong coreset.
        :return: Khoảng cách gradient chuẩn hóa (float); trả về 0.0 nếu coreset rỗng.
        """
        if not coreset_indices or not coreset_weights:
            return 0.0
        total_full_gradient = None
        total_coreset_gradient = None
        for c in range(self.num_classes):
            class_indices = self.indices_by_class.get(c, [])
            if len(class_indices) == 0:
                continue
            class_gradients = compute_gradient_representations(
                model=self.model,
                full_dataset=self.full_dataset,
                indices=class_indices,
                device=self.device,
                gradient_type=self.gradient_type,
                batch_size=self.batch_size,
            )
            if class_gradients.shape[0] == 0:
                continue
            if total_full_gradient is None:
                grad_dim = class_gradients.shape[1]
                total_full_gradient = torch.zeros(grad_dim, dtype=torch.float32)
                total_coreset_gradient = torch.zeros(grad_dim, dtype=torch.float32)
            total_full_gradient += class_gradients.sum(dim=0)
            global_to_local = {
                g_idx: i for i, g_idx in enumerate(class_indices)
            }
            class_coreset_local = []
            class_weights = []
            for g_idx in coreset_indices:
                if g_idx in global_to_local:
                    class_coreset_local.append(global_to_local[g_idx])
                    class_weights.append(float(coreset_weights.get(g_idx, 1.0)))
            if len(class_coreset_local) == 0:
                continue
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            coreset_class_gradient = torch.sum(
                class_gradients[class_coreset_local] * weights_tensor.unsqueeze(1),
                dim=0,
            )
            total_coreset_gradient += coreset_class_gradient
        if total_full_gradient is None:
            return 0.0
        gradient_distance = torch.norm(total_full_gradient - total_coreset_gradient).item()
        full_gradient_norm = torch.norm(total_full_gradient).item()
        normalized_distance = gradient_distance / (full_gradient_norm + 1e-8)
        return normalized_distance

class WeightedSubsetDataset(Dataset):
    """
    Lớp Dataset bọc quanh full_dataset, chỉ chứa các mẫu thuộc coreset kèm theo trọng số tương ứng.
    """
    def __init__(self, full_dataset, indices, weights):
        """
        Hàm khởi tạo, lưu lại tập dữ liệu gốc cùng danh sách chỉ số và trọng số của coreset.
        :param full_dataset: Toàn bộ tập dữ liệu gốc.
        :param indices: Danh sách chỉ số toàn cục của các mẫu thuộc coreset.
        :param weights: Dict ánh xạ chỉ số toàn cục sang trọng số của mẫu trong coreset.
        """
        self.full_dataset = full_dataset
        self.indices = indices
        self.weights = weights
    def __len__(self):
        """
        Hàm trả về số lượng mẫu trong coreset.
        :return: Số lượng phần tử trong self.indices.
        """
        return len(self.indices)
    def __getitem__(self, idx):
        """
        Hàm lấy một mẫu (dữ liệu, nhãn, trọng số) theo chỉ số cục bộ trong coreset.
        :param idx: Chỉ số cục bộ (vị trí trong self.indices) của mẫu cần lấy.
        :return: Bộ ba (x, y, weight) gồm dữ liệu đầu vào, nhãn, và trọng số (dạng tensor).
        """
        global_idx = self.indices[idx]
        x, y = self.full_dataset[global_idx]
        weight = self.weights.get(global_idx, 1.0)
        return x, y, torch.tensor(weight, dtype=torch.float32)