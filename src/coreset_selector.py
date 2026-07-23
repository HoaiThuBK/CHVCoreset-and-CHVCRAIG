"""
---file coreset_selector.py---

Dispatcher for coreset selection across multiple methods. Provides a `CoresetSelector` class that routes to the different selection 
strategies (CRAIG, CHVCoreset, Random, CHV-CRAIG) behind a single `select` interface, plus a utility to evaluate coreset quality via the normalized
gradient distance between the weighted coreset gradient and the full-dataset gradient. Also provides `set_seed` for reproducibility and a
`WeightedSubsetDataset` that exposes the selected coreset (with per-sample weights) as a torch Dataset.
"""
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from utils import compute_gradient_representations
from random_selector import select_random_coreset
from craig import select_craig_coreset
from chvcoreset import select_chvcoreset
from chv_craig import select_chv_craig_coreset


def set_seed(seed: int):
    """
    Set the seed for the random number generators (random, numpy, torch, cudnn) to make results reproducible.

    :param seed: Seed value used to initialize the random number generators.
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
    Dispatcher class for coreset selection across several different methods (craig, chvcoreset, random, chv_craig).
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
        Initialize and store the parameters shared by all coreset selection methods.

        :param model: Model used to compute the per-sample gradients.
        :param full_dataset: The complete original dataset.
        :param indices_by_class: Dict mapping a class label to a list of indices into full_dataset.
        :param coreset_size_fraction: Coreset size as a fraction of full_dataset.
        :param device: Device (CPU/GPU) used when computing gradients.
        :param gradient_type: Type of gradient representation to compute.
        :param num_classes: Total number of classes in the dataset.
        :param candidate_multiplier: Multiplier that controls the number of candidates, used by the craig_ch (chv_craig) method.
        :param random_state: Seed for the randomized steps in the coreset selection process.
        :param batch_size: Batch size used when computing gradients.
        :param chvs4_epsilon: Epsilon stopping threshold used by the chvcoreset/chv_craig methods.
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
        Select a coreset using the specified method.

        :param method: Name of the coreset selection method ("craig", "chvcoreset", "random", "chv_craig").
        :return: The value returned by the select function corresponding to the chosen method.
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
        
        elif method == "chvcoreset":
            return select_chvcoreset(
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
        elif method == "chv_craig":
            return select_chv_craig_coreset(
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
            raise ValueError(f"Invalid selection method: {method}")
        
    def compute_grad_dist_for_current_coreset(self, coreset_indices, coreset_weights):
        """
        Compute the normalized gradient distance between the weighted total gradient of the coreset and the total gradient of the full dataset, used to assess coreset quality.

        :param coreset_indices: List of global indices of the samples in the coreset.
        :param coreset_weights: Dict mapping a global index to the weight of the corresponding sample in the coreset.
        :return: The normalized gradient distance (float); returns 0.0 if the coreset is empty.
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
    Dataset wrapper around full_dataset that contains only the samples belonging to the coreset, together with their corresponding weights.
    """
    def __init__(self, full_dataset, indices, weights):
        """
        Initialize and store the original dataset along with the coreset's list of indices and weights.

        :param full_dataset: The complete original dataset.
        :param indices: List of global indices of the samples belonging to the coreset.
        :param weights: Dict mapping a global index to the weight of the corresponding sample in the coreset.
        """
        self.full_dataset = full_dataset
        self.indices = indices
        self.weights = weights

    def __len__(self):
        """
        Return the number of samples in the coreset.

        :return: The number of elements in self.indices.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get one sample (data, label, weight) by its local index within the coreset.

        :param idx: Local index (position in self.indices) of the sample to retrieve.
        :return: A tuple (x, y, weight) of the input data, the label, and the weight (as a tensor).
        """
        global_idx = self.indices[idx]
        x, y = self.full_dataset[global_idx]
        weight = self.weights.get(global_idx, 1.0)
        return x, y, torch.tensor(weight, dtype=torch.float32)