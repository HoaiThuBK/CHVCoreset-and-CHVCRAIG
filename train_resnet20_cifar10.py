"""
---file train_resnet20_cifar10.py---
Script huấn luyện mô hình ResNet-20 trên CIFAR-10 (10 lớp), có hỗ trợ nhiều phương pháp chọn
coreset (craig, chvs4, random, craig_ch) hoặc huấn luyện trên toàn bộ dataset.
"""
import numpy as np
import torch
import csv 
import os
import argparse
import random
import torch.nn as nn
import torch.optim as optim
from utils import load_cifar10_all, get_indices_by_class, train_one_epoch, evaluate

from model_resnet import resnet20
from coreset_selector import CoresetSelector, WeightedSubsetDataset

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

def main():
    """
    Hàm chính: đọc tham số dòng lệnh, chuẩn bị dữ liệu/mô hình/optimizer, sau đó chạy vòng lặp
    huấn luyện qua từng epoch (warm-up trên full dataset, rồi định kỳ chọn lại coreset và huấn
    luyện có trọng số), đồng thời ghi lại kết quả (loss, accuracy, thời gian...) ra file CSV.
    """
    parser = argparse.ArgumentParser(description='Huấn luyện Coreset với Resnet20 trên CIFAR 10 lớp')

    # --- Định nghĩa các tham số dòng lệnh ---
    parser.add_argument('--selection_method', type=str, default='full_dataset', 
                        choices=['craig', 'chvs4', 'full_dataset', 'random','craig_ch'], 
                        help='Phương pháp lựa chọn coreset hoặc huấn luyện đầy đủ')
    parser.add_argument('--coreset_fraction', type=float, default=0.01, 
                        help='Tỷ lệ coreset mong muốn (ví dụ: 0.1, 0.3)')
    parser.add_argument('--update_freq', type=int, default=5, 
                        help='Tần suất cập nhật coreset (số epoch). Mặc định: 5')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Tổng số epoch để huấn luyện. Mặc định: 100')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='Tốc độ học ban đầu. Mặc định: 0.1')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='Kích thước mini-batch. Mặc định: 512')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Số epoch để khởi động tốc độ học. Mặc định: 5')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gradient_type', type=str, default="logit", choices=["logit", "embedding"],
                        help='Phương pháp biểu diễn gradient. Mặc định là logit gradient')
    parser.add_argument('--candidate_multiplier', type=int, default=3,
                        help='Hệ số quy định số candidate_budget (candidate_budget = candidate_multiplier * budget). Mặc định : 3')
    parser.add_argument('--coreset_lr_scale', type=float, default=0.1,
                    help='Hệ số giảm LR khi chuyển sang train bằng coreset. Ví dụ 0.1: LR 0.1 -> 0.01')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='Hệ số giảm Learning Rate tại 50% và 75% số epoch')
    
    args = parser.parse_args()
    set_seed(args.seed)
    print(f"Sử dụng seed: {args.seed}")


    # --- Thiết lập môi trường (ưu tiên MPS > CUDA > CPU) ---
    if torch.backends.mps.is_available(): device = 'mps'
    elif torch.cuda.is_available(): device = 'cuda'
    else: device = 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    # --- Load dữ liệu CIFAR-10 ---
    trainset_full, testset = load_cifar10_all()
    trainset_for_selection = trainset_full
    num_classes = 10
    input_dim = None
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Khởi tạo mô hình ---
    print(f"Sử dụng mô hình: Resnet20 ({input_dim} features)")
    model = resnet20(num_classes=10).to(device)

    # --- Lấy indices theo lớp (dùng cho các phương pháp chọn coreset theo từng lớp) ---
    indices_by_class = get_indices_by_class(trainset_for_selection, num_classes=10)

    # --- Optimizer (Chỉ dùng SGD cho thí nghiệm này) ---
    l2_reg = 1e-5 
    print(f"Sử dụng Optimizer: SGD (LR={args.lr}, L2 Reg={l2_reg})")
    criterion = nn.CrossEntropyLoss(reduction='none') 
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=l2_reg)

    def get_current_lr(epoch, args):
        """
        Hàm tính learning rate hiện tại theo epoch: warm-up tuyến tính, sau đó giảm LR nếu train
        bằng coreset, và giảm tiếp tại 50%/75% tổng số epoch (thay cho torch LR scheduler).
        :param epoch: Chỉ số epoch hiện tại (bắt đầu từ 0).
        :param args: Namespace tham số dòng lệnh (chứa lr, warmup_epochs, epochs, ...).
        :return: Giá trị learning rate áp dụng cho epoch hiện tại.
        """
        current_epoch = epoch + 1

        # Warm-up: tăng tuyến tính LR
        if current_epoch <= args.warmup_epochs:
            return args.lr * current_epoch / args.warmup_epochs

        # Sau warm-up: nếu train bằng coreset thì giảm LR
        lr = args.lr

        if args.selection_method != "full_dataset":
            lr = lr * args.coreset_lr_scale

        # Giảm LR tại 50% và 75% tổng số epoch
        milestone_1 = int(args.epochs * 0.5)
        milestone_2 = int(args.epochs * 0.75)

        if current_epoch >= milestone_2:
            lr = lr * (args.lr_gamma ** 2)
        elif current_epoch >= milestone_1:
            lr = lr * args.lr_gamma

        return lr

    # --- Khởi tạo bộ chọn coreset (dùng chung cho mọi phương pháp selection_method) ---
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

    # --- Thiết lập lưu file .csv ---
    output_dir = os.path.join("results_cifar10", args.selection_method)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"results_cifar10_resnet_{args.selection_method}_{args.gradient_type}_seed{args.seed}"

    if args.selection_method != 'full_dataset':
        filename += f"_frac{args.coreset_fraction}"
    filename += ".csv"
    filepath = os.path.join(output_dir, filename)
    print(f"Sẽ lưu kết quả vào: {filepath}")
    csv_file = open(filepath, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'loss', 'accuracy', 'lr', 'train_time_s', 'selection_time_s','coreset_size'])

    coreset_indices, coreset_weights = None, None
    current_coreset_size = 0
    g = torch.Generator()
    g.manual_seed(args.seed)
    full_trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    # --- Vòng lặp huấn luyện chính ---
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        current_lr = get_current_lr(epoch, args)

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        print(f">>> Current LR: {current_lr:.6f}")
        selection_time = 0.0

        if args.selection_method == 'full_dataset':
            # Chế độ baseline: luôn huấn luyện trên toàn bộ dataset, không chọn coreset.
            print("Huấn luyện trên toàn bộ dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)
        
        elif epoch < args.warmup_epochs:
            # Giai đoạn warm-up: luôn dùng toàn bộ dataset để mô hình ổn định trước khi chọn coreset.
            print(f"Giai đoạn Warm-up (Epoch {epoch+1}/{args.warmup_epochs}): Huấn luyện trên toàn bộ dataset...")
            trainloader = full_trainloader
            use_weights_in_loss = False
            selection_time = 0.0
            current_coreset_size = len(trainset_full)
        else:
            # Sau warm-up: huấn luyện có trọng số, định kỳ (update_freq epoch) chọn lại coreset.
            use_weights_in_loss = True
            if (epoch - args.warmup_epochs) % args.update_freq == 0:
                print(f"Đã qua Warm-up. Cập nhật coreset tại epoch {epoch+1}...")
                result = selector.select(method=args.selection_method)
                # Một số phương pháp (vd craig_ch) trả về thêm 1 giá trị phụ, chỉ lấy 3 giá trị cần dùng.
                if len(result) == 4:
                    coreset_indices, coreset_weights, selection_time, _ = result
                else:
                    coreset_indices, coreset_weights, selection_time = result
                current_coreset_size = len(coreset_indices) if coreset_indices is not None else 0

                print(f"Số lượng điểm trong coreset: {current_coreset_size}")

                if not coreset_indices: 
                    # Coreset rỗng: ghi log epoch này với giá trị 0 rồi bỏ qua, không huấn luyện.
                    print("Cảnh báo: Coreset rỗng, bỏ qua epoch này.")
                    csv_writer.writerow([epoch+1, 0, 0, optimizer.param_groups[0]['lr'], 0, selection_time, current_coreset_size])
                    continue
            else:
                # Không đến kỳ cập nhật: tái sử dụng coreset đã chọn ở epoch trước đó.
                selection_time = 0.0 

            if not coreset_indices:
                 # Chưa từng chọn được coreset hợp lệ: tạm thời fallback về full dataset.
                 print("Cảnh báo: coreset_indices rỗng, dùng full dataset cho epoch này.")
                 trainloader = full_trainloader
                 use_weights_in_loss = False
            else:
                coreset_dataset = WeightedSubsetDataset(trainset_full, coreset_indices, coreset_weights)
                
                g_coreset = torch.Generator()
                g_coreset.manual_seed(args.seed)
                trainloader = torch.utils.data.DataLoader(coreset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g_coreset)

        # --- Huấn luyện 1 epoch và đánh giá trên tập test ---
        train_loss, train_time = train_one_epoch(model, trainloader, criterion, optimizer, device, use_weights=use_weights_in_loss)
        accuracy = evaluate(model, testloader, device)  
        current_lr = optimizer.param_groups[0]['lr'] 

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={accuracy:.2f}%, LR={current_lr:.5f}, Time(Train)={train_time:.2f}s, Time(Select)={selection_time:.2f}s")

        csv_writer.writerow([epoch+1, f"{train_loss:.4f}", f"{accuracy:.2f}", f"{current_lr:.5f}", f"{train_time:.2f}", f"{selection_time:.2f}", current_coreset_size])
        csv_file.flush() 

    csv_file.close()
    print(f"\nHuấn luyện hoàn tất. Kết quả đã được lưu vào file: {filename}")


if __name__ == '__main__':
    main()