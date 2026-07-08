"""
---file utils.py---
Chứa một số hàm chuẩn bị cơ bản:
- Load dữ liệu FashionMNIST 10 lớp
- Load dữ liệu CIFAR-10 
- Hàm lấy indices cho từng lớp
- Hàm biểu diễn gradient
- Định nghĩa hàm train mô hình trên một epoch
- Hàm đánh giá độ chính xác 
"""

from __future__ import annotations
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def load_fashion_mnist_all(data_root: str = "./data_mnist"):
    """
    Hàm tải tập FashionMNIST (train + test), chuẩn hóa và flatten ảnh thành vector 784 chiều.
    :param data_root: Thư mục lưu/tải dữ liệu FashionMNIST.
    :return: Bộ đôi (trainset, testset) là hai đối tượng Dataset của torchvision.
    """
    print(">>> Loading FashionMNIST with 10 classes...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    trainset = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    print(f"Train size: {len(trainset)}")
    print(f"Test size : {len(testset)}")
    print("Classes   : 10")
    return trainset, testset

def load_cifar10_all(data_root="./data_cifar10"):
    """
    Hàm tải tập CIFAR-10 (train + test), có augmentation (crop, flip) cho tập train.
    :param data_root: Thư mục lưu/tải dữ liệu CIFAR-10.
    :return: Bộ đôi (trainset, testset) là hai đối tượng Dataset của torchvision.
    """
    print(">>> Loading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )
    print(f"Train size: {len(trainset)}")
    print(f"Test size : {len(testset)}")
    print("Classes   : 10")
    return trainset, testset

def load_svhn_all(data_root="./data_svhn"):
    """
    Hàm tải tập SVHN (train + test), có augmentation crop cho tập train.
    :param data_root: Thư mục lưu/tải dữ liệu SVHN.
    :return: Bộ đôi (trainset, testset) là hai đối tượng Dataset của torchvision.
    """
    print(">>> Loading SVHN...")
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4377, 0.4438, 0.4728),
            std=(0.1980, 0.2010, 0.1970)
        )
    ])
    trainset = torchvision.datasets.SVHN(
        root=data_root,
        split='train',
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.SVHN(
        root=data_root,
        split='test',
        download=True,
        transform=transform_test
    )

    print(f"Train size: {len(trainset)}")
    print(f"Test size : {len(testset)}")
    print("Classes   : 10")
    return trainset, testset

@torch.no_grad()
def dataset_to_tensors(dataset, batch_size: int = 2048, device: str | torch.device = "cpu"):
    """
    Hàm gộp toàn bộ một Dataset thành hai tensor (X, y) bằng cách duyệt qua DataLoader.
    :param dataset: Dataset cần chuyển thành tensor.
    :param batch_size: Kích thước batch khi duyệt qua dataset.
    :param device: Thiết bị (CPU/GPU) chứa tensor kết quả.
    :return: Bộ đôi (X, y) là tensor dữ liệu (float32) và tensor nhãn (long).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    X_list, y_list = [], []
    for xb, yb in loader:
        X_list.append(xb)
        y_list.append(yb)
    X = torch.cat(X_list, dim=0).to(device=device, dtype=torch.float32)
    y = torch.cat(y_list, dim=0).to(device=device, dtype=torch.long)
    return X, y

def get_indices_by_class(full_dataset, num_classes):
    """
    Hàm nhóm chỉ số các mẫu trong full_dataset theo từng lớp nhãn.
    :param full_dataset: Tập dữ liệu cần lấy chỉ số theo lớp (có targets/labels/tensors hoặc iterable).
    :param num_classes: Tổng số lớp trong tập dữ liệu.
    :return: Dict ánh xạ nhãn lớp sang danh sách chỉ số mẫu thuộc lớp đó.
    """
    indices = {c: [] for c in range(num_classes)}

    if hasattr(full_dataset, "targets"):
        labels = full_dataset.targets
    elif hasattr(full_dataset, "labels"):
        labels = full_dataset.labels
    elif hasattr(full_dataset, "tensors"):
        labels = full_dataset.tensors[1]
    else:
        labels = [label for _, label in full_dataset]

    for idx, label in enumerate(labels):
        if isinstance(label, torch.Tensor):
            label = label.item()
        indices[int(label)].append(idx)

    return indices

@torch.no_grad()
def compute_gradient_representations(
    model,
    full_dataset,
    indices,
    device,
    gradient_type,
    batch_size,
    return_device="cpu"
):
    """
    Hàm tính biểu diễn gradient cho từng mẫu, dựa trên chênh lệch giữa xác suất dự đoán (softmax) và nhãn one-hot, theo kiểu "logit" hoặc "embedding".
    :param model: Mô hình dùng để tính logits/embedding cho từng mẫu.
    :param full_dataset: Toàn bộ tập dữ liệu gốc.
    :param indices: Danh sách chỉ số các mẫu (trong full_dataset) cần tính gradient.
    :param device: Thiết bị (CPU/GPU) dùng để chạy mô hình.
    :param gradient_type: Loại biểu diễn gradient, "logit" hoặc "embedding".
    :param batch_size: Kích thước batch khi duyệt qua các mẫu.
    :param return_device: Thiết bị chứa tensor kết quả, "cpu" hoặc "device".
    :return: Tensor gradient của các mẫu trong indices, kích thước (len(indices), feature_dim).
        Trả về tensor rỗng nếu indices rỗng.
    """
    if len(indices) == 0:
        out_device = device if return_device == "device" else "cpu"
        return torch.empty(0, 0, device=out_device)
    gradient_type = gradient_type.lower()
    model.eval()
    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    grads = []
    for xb, yb in loader:
        xb = xb.to(device=device, dtype=torch.float32)
        yb = yb.to(device)
        logits = model(xb)
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(yb, num_classes=num_classes).to(device=device, dtype=torch.float32)
        logit_grad = probs - one_hot
        if gradient_type == "logit":
            batch_grad = logit_grad
        elif gradient_type == "embedding":
            z = model.embedding(xb)
            batch_grad = (logit_grad.unsqueeze(2) * z.unsqueeze(1)).flatten(1)
        else:
            raise ValueError("gradient_type must be 'logit' or 'embedding'")
        if return_device == "cpu":
            grads.append(batch_grad.detach().cpu())
        elif return_device == "device":
            grads.append(batch_grad.detach())
        else:
            raise ValueError("return_device must be 'cpu' or 'device'")
    return torch.cat(grads, dim=0)

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    use_weights=True,
    grad_clip_norm=None):
    """
    Hàm huấn luyện mô hình qua một epoch, hỗ trợ loss có trọng số theo mẫu (weighted coreset).
    :param model: Mô hình cần huấn luyện.
    :param dataloader: DataLoader cung cấp batch dữ liệu (có hoặc không kèm trọng số).
    :param criterion: Hàm loss, trả về loss theo từng mẫu (reduction='none').
    :param optimizer: Optimizer dùng để cập nhật tham số mô hình.
    :param device: Thiết bị (CPU/GPU) dùng để huấn luyện.
    :param use_weights: Nếu True, dataloader trả về thêm trọng số mẫu và loss được tính có trọng số.
    :param grad_clip_norm: Ngưỡng clip gradient theo norm; None nghĩa là không clip.
    :return: Bộ đôi (avg_loss, train_time) — loss trung bình mỗi batch và thời gian huấn luyện (giây).
    """
    model.train()
    total_loss = 0.0
    if len(dataloader) == 0:
        return 0.0, 0.0
    start_time = time.time()
    for batch in dataloader:
        if use_weights:
            images, labels, weights = batch
            weights = weights.to(device=device, dtype=torch.float32)
        else:
            images, labels = batch
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        per_sample_loss = criterion(outputs, labels)
        if use_weights:
            weights_sum = weights.sum()
            if weights_sum > 0:
                loss = (per_sample_loss * weights).sum() / weights_sum
            else:
                loss = per_sample_loss.mean()
        else:
            loss = per_sample_loss.mean()
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
    train_time = time.time() - start_time
    return total_loss / len(dataloader), train_time

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Hàm đánh giá độ chính xác (accuracy) của mô hình trên một dataloader.
    :param model: Mô hình cần đánh giá.
    :param dataloader: DataLoader cung cấp batch dữ liệu (ảnh, nhãn) để đánh giá.
    :param device: Thiết bị (CPU/GPU) dùng để chạy mô hình.
    :return: Độ chính xác theo phần trăm (0.0 nếu dataloader rỗng).
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total

def calculate_weights(class_gradients, selected_local):
    """
    Hàm tính trọng số cho từng mẫu được chọn, bằng cách gán mỗi mẫu trong class_gradients cho
    mẫu được chọn gần nhất (theo khoảng cách Euclid) rồi đếm số mẫu được gán cho từng mẫu chọn.
    :param class_gradients: Ma trận gradient của tất cả các mẫu trong lớp, kích thước (n, dim).
    :param selected_local: Danh sách chỉ số cục bộ các mẫu được chọn vào coreset.
    :return: Dict ánh xạ chỉ số cục bộ mẫu được chọn sang trọng số (số mẫu được gán cho nó).
        Trả về dict rỗng nếu selected_local rỗng.
    """
    if len(selected_local) == 0:
        return {}
    selected_local = list(dict.fromkeys(selected_local))
    n = class_gradients.shape[0]
    if len(selected_local) == 1:
        return {selected_local[0]: float(n)}
    selected_gradients = class_gradients[selected_local]
    dists = torch.cdist(class_gradients, selected_gradients, p=2) ** 2
    closest = torch.argmin(dists, dim=1)
    weights = {idx: 0.0 for idx in selected_local}
    for pos in closest.tolist():
        local_idx = selected_local[pos]
        weights[local_idx] += 1.0
    return weights