"""
--- file utils.py ---

Contains a few basic preparation utilities:
- Load the FashionMNIST dataset (10 classes)
- Load the CIFAR-10 dataset
- Function to get the indices of each class
- Gradient representation function
- Definition of the function that trains the model for one epoch
- Accuracy evaluation function
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
    Load the FashionMNIST dataset (train + test), normalize it, and flatten each image into a 784-dimensional vector.

    :param data_root: Directory where the FashionMNIST data is stored/downloaded.
    :return: A pair (trainset, testset) of two torchvision Dataset objects.
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
    Load the CIFAR-10 dataset (train + test), with augmentation (crop, flip) for the train set.

    :param data_root: Directory where the CIFAR-10 data is stored/downloaded.
    :return: A pair (trainset, testset) of two torchvision Dataset objects.
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
    Load the SVHN dataset (train + test), with crop augmentation for the train set.

    :param data_root: Directory where the SVHN data is stored/downloaded.
    :return: A pair (trainset, testset) of two torchvision Dataset objects.
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
    Collapse an entire Dataset into two tensors (X, y) by iterating over a DataLoader.

    :param dataset: The dataset to convert into tensors.
    :param batch_size: Batch size used when iterating over the dataset.
    :param device: Device (CPU/GPU) that holds the resulting tensors.
    :return: A pair (X, y) of the data tensor (float32) and the label tensor (long).
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
    Group the indices of the samples in full_dataset by their class label.

    :param full_dataset: The dataset from which to get indices by class (having targets/labels/tensors, or otherwise iterable).
    :param num_classes: Total number of classes in the dataset.
    :return: Dict mapping a class label to the list of indices of the samples belonging to that class.
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
    Compute a gradient representation for each sample, based on the difference between the predicted probabilities (softmax) and the one-hot label, in
    either "logit" or "embedding" form.

    :param model: Model used to compute the logits/embedding for each sample.
    :param full_dataset: The complete original dataset.
    :param indices: List of indices of the samples (in full_dataset) whose gradient is to be computed.
    :param device: Device (CPU/GPU) used to run the model.
    :param gradient_type: Type of gradient representation, "logit" or "embedding".
    :param batch_size: Batch size used when iterating over the samples.
    :param return_device: Device that holds the resulting tensor, "cpu" or "device".
    :return: Gradient tensor of the samples in indices, of shape (len(indices), feature_dim). Returns an empty tensor if indices is empty.
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
    Train the model for one epoch, supporting a per-sample weighted loss (weighted coreset).

    :param model: The model to train.
    :param dataloader: DataLoader providing the data batches (with or without weights).
    :param criterion: Loss function that returns the per-sample loss (reduction='none').
    :param optimizer: Optimizer used to update the model parameters.
    :param device: Device (CPU/GPU) used for training.
    :param use_weights: If True, the dataloader also returns per-sample weights and the loss is computed with weights.
    :param grad_clip_norm: Gradient clipping threshold by norm; None means no clipping.
    :return: A pair (avg_loss, train_time) — the average loss per batch and the training time (in seconds).
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
    Evaluate the accuracy of the model on a dataloader.

    :param model: The model to evaluate.
    :param dataloader: DataLoader providing the data batches (images, labels) for evaluation.
    :param device: Device (CPU/GPU) used to run the model.
    :return: The accuracy as a percentage (0.0 if the dataloader is empty).
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
    Compute the weight of each selected sample by assigning every sample in class_gradients to its nearest selected sample (by Euclidean distance) and
    then counting how many samples are assigned to each selected sample.

    :param class_gradients: Gradient matrix of all samples in the class, of shape (n, dim).
    :param selected_local: List of local indices of the samples selected into the coreset.
    :return: Dict mapping the local index of a selected sample to its weight (the number of samples assigned to it). Returns an empty dict if selected_local is empty.
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