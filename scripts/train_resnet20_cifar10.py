"""
---file train_resnet20_cifar10.py---

Training script for a ResNet-20 model on CIFAR-10 (10 classes), with support for several coreset selection methods (craig, chvcoreset, random, chv_craig) or training on the full dataset.
"""
import numpy as np
import torch
import csv 
import os
import sys
import argparse
import random
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import load_cifar10_all, get_indices_by_class, train_one_epoch, evaluate
from model_resnet import resnet20
from coreset_selector import CoresetSelector, WeightedSubsetDataset

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

def main():
    """
    Main function: parse command-line arguments, prepare the data/model/optimizer, then run the training loop over epochs (warm-up on the full dataset, then
    periodically re-select the coreset and train with weights), while logging the results (loss, accuracy, time, ...) to a CSV file.
    """
    parser = argparse.ArgumentParser(description='Coreset training with ResNet-20 on CIFAR-10 (10 classes)')

    # --- Define the command-line arguments ---
    parser.add_argument('--selection_method', type=str, default='full_dataset', 
                        choices=['craig', 'chvcoreset', 'full_dataset', 'random','chv_craig'], 
                        help='Coreset selection method, or full-dataset training')
    parser.add_argument('--coreset_fraction', type=float, default=0.01, 
                        help='Desired coreset fraction (e.g. 0.1, 0.3)')
    parser.add_argument('--update_freq', type=int, default=50, 
                        help='Coreset update frequency (in epochs).')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Total number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Mini-batch size.')
    parser.add_argument('--warmup_epochs', type=int, default=20, 
                        help='Number of warm-up epochs for the learning rate.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gradient_type', type=str, default="logit", choices=["logit", "embedding"],
                        help='Gradient representation method. Default is logit gradient')
    parser.add_argument('--candidate_multiplier', type=int, default=5,
                        help='Factor determining candidate_budget (candidate_budget = candidate_multiplier * budget). Default: 5')
    parser.add_argument('--coreset_lr_scale', type=float, default=0.1,
                    help='LR reduction factor when switching to coreset training. E.g. 0.1: LR 0.1 -> 0.01')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='Learning rate decay factor at 50% and 75% of the epochs')
    
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Using seed: {args.seed}")


    # --- Set up the environment  ---
    if torch.backends.mps.is_available(): device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print(f"Using device: {device}")
    
    # --- Load the CIFAR-10 data ---
    trainset_full, testset = load_cifar10_all()
    trainset_for_selection = trainset_full
    num_classes = 10
    input_dim = None
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Initialize the model ---
    print(f"Using model: ResNet-20")
    model = resnet20(num_classes=10).to(device)

    # --- Get indices by class (used by the per-class coreset selection methods) ---
    indices_by_class = get_indices_by_class(trainset_for_selection, num_classes=10)

    # --- Optimizer (Only SGD is used for this experiment) ---
    l2_reg = 1e-5 
    print(f"Using Optimizer: SGD (LR={args.lr}, L2 Reg={l2_reg})")
    criterion = nn.CrossEntropyLoss(reduction='none') 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=l2_reg)

    def get_current_lr(epoch, args):
        """
        Compute the current learning rate as a function of the epoch: linear warm-up, then reduce the LR if training on a coreset, and reduce it
        further at 50%/75% of the total number of epochs (in place of a torch LR scheduler).

        :param epoch: Current epoch index (starting from 0).
        :param args: Command-line argument namespace (contains lr, warmup_epochs, epochs, ...).
        :return: The learning rate value to apply for the current epoch.
        """
        current_epoch = epoch + 1

        # Warm-up: increase the LR linearly
        if current_epoch <= args.warmup_epochs:
            return args.lr * current_epoch / args.warmup_epochs

        # After warm-up: reduce the LR if training on a coreset
        lr = args.lr

        if args.selection_method != "full_dataset":
            lr = lr * args.coreset_lr_scale

        # Reduce the LR at 50% and 75% of the total number of epochs
        milestone_1 = int(args.epochs * 0.5)
        milestone_2 = int(args.epochs * 0.75)

        if current_epoch >= milestone_2:
            lr = lr * (args.lr_gamma ** 2)
        elif current_epoch >= milestone_1:
            lr = lr * args.lr_gamma

        return lr

    # --- Initialize the coreset selector (shared by every selection_method) ---
    selector = CoresetSelector(
        model=model,
        full_dataset=trainset_for_selection,
        indices_by_class=indices_by_class,
        coreset_size_fraction=args.coreset_fraction,
        device=device,
        gradient_type=args.gradient_type,
        batch_size=args.batch_size,
        num_classes=num_classes,
        candidate_multiplier = args.candidate_multiplier,
        random_state = args.seed
    )

    # --- Set up saving to the .csv file ---
    output_dir = os.path.join("results_cifar10", args.selection_method)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"results_cifar10_resnet_{args.selection_method}_{args.gradient_type}_seed{args.seed}"

    if args.selection_method != 'full_dataset':
        filename += f"_frac{args.coreset_fraction}"
    filename += ".csv"
    filepath = os.path.join(output_dir, filename)
    print(f"Results will be saved to: {filepath}")
    csv_file = open(filepath, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'loss', 'accuracy', 'lr', 'train_time_s', 'selection_time_s','coreset_size'])

    coreset_indices, coreset_weights = None, None
    current_coreset_size = 0
    g = torch.Generator()
    g.manual_seed(args.seed)
    full_trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    # --- Main training loop ---
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        current_lr = get_current_lr(epoch, args)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        print(f">>> Current LR: {current_lr:.6f}")
        selection_time = 0.0

        if args.selection_method == 'full_dataset':
            # Baseline mode: always train on the full dataset, no coreset selection.
            print("Training on the full dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)
        
        elif epoch < args.warmup_epochs:
            # Warm-up phase: always use the full dataset so the model stabilizes before selecting a coreset.
            print(f"Warm-up phase (Epoch {epoch+1}/{args.warmup_epochs}): Training on the full dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)
        else:
            # After warm-up: weighted training, periodically (every update_freq epochs) re-select the coreset.
            use_weights_in_loss = True
            if (epoch - args.warmup_epochs) % args.update_freq == 0:
                print(f"Warm-up complete. Updating coreset at epoch {epoch+1}...")
                result = selector.select(method=args.selection_method)
                # Some methods (e.g. chv_craig) return an extra value; keep only the 3 values we need.
                if len(result) == 4:
                    coreset_indices, coreset_weights, selection_time, _ = result
                else:
                    coreset_indices, coreset_weights, selection_time = result
                current_coreset_size = len(coreset_indices) if coreset_indices is not None else 0

                print(f"Number of points in the coreset: {current_coreset_size}")

                if not coreset_indices: 
                    # Empty coreset: log this epoch with zero values and skip it, no training.
                    print("Warning: Coreset is empty, skipping this epoch.")
                    csv_writer.writerow([epoch+1, 0, 0, optimizer.param_groups[0]['lr'], 0, selection_time, current_coreset_size])
                    continue
            else:
                # Not an update epoch: reuse the coreset selected in a previous epoch.
                selection_time = 0.0 

            if not coreset_indices:
                 # No valid coreset has ever been selected: temporarily fall back to the full dataset.
                 print("Warning: coreset_indices is empty, using the full dataset for this epoch.")
                 trainloader = full_trainloader
                 use_weights_in_loss = False
            else:
                coreset_dataset = WeightedSubsetDataset(trainset_full, coreset_indices, coreset_weights)
                
                g_coreset = torch.Generator()
                g_coreset.manual_seed(args.seed)
                trainloader = torch.utils.data.DataLoader(coreset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g_coreset)

        # --- Train for one epoch and evaluate on the test set ---
        train_loss, train_time = train_one_epoch(model, trainloader, criterion, optimizer, device, use_weights=use_weights_in_loss)
        accuracy = evaluate(model, testloader, device)  
        current_lr = optimizer.param_groups[0]['lr'] 

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, LR={current_lr:.5f}, Time(Train)={train_time:.2f}s, Time(Select)={selection_time:.2f}s")

        csv_writer.writerow([epoch+1, f"{train_loss:.4f}", f"{accuracy:.2f}", f"{current_lr:.5f}", f"{train_time:.2f}", f"{selection_time:.2f}", current_coreset_size])
        csv_file.flush() 

    csv_file.close()
    print(f"\nTraining complete. Results have been saved to file: {filename}")


if __name__ == '__main__':
    main()