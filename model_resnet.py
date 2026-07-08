"""
---file resnet.py---
Định nghĩa mô hình Resnet-20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _weights_init(m):
    """
    Hàm khởi tạo trọng số cho các layer Conv2d/Linear (Kaiming normal) và BatchNorm2d (weight=1, bias=0).
    :param m: Module (layer) của mô hình cần khởi tạo trọng số.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class BasicBlock(nn.Module):
    """
    Khối residual cơ bản của ResNet gồm 2 lớp conv 3x3 kèm BatchNorm và kết nối tắt (shortcut).
    """
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        """
        Hàm khởi tạo khối BasicBlock, tự tạo shortcut projection nếu số kênh/stride thay đổi.
        :param in_planes: Số kênh đầu vào của khối.
        :param planes: Số kênh đầu ra của khối.
        :param stride: Bước trượt (stride) của lớp conv đầu tiên.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    def forward(self, x):
        """
        Hàm lan truyền xuôi qua khối BasicBlock, cộng kết nối tắt rồi qua ReLU.
        :param x: Tensor đầu vào, kích thước (batch, in_planes, H, W).
        :return: Tensor đầu ra sau khối residual, kích thước (batch, planes, H', W').
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    """
    Mô hình ResNet dạng CIFAR (3 stage, mỗi stage gồm nhiều BasicBlock) dùng cho ảnh 32x32.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Hàm khởi tạo mô hình ResNet, xây 3 stage từ block cơ bản rồi khởi tạo trọng số.
        :param block: Lớp block dùng để xây các stage (ví dụ BasicBlock).
        :param num_blocks: Danh sách số lượng block trong mỗi stage (3 stage).
        :param num_classes: Số lớp đầu ra của mô hình.
        """
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Hàm tạo một stage gồm nhiều block liên tiếp, chỉ block đầu tiên dùng stride truyền vào.
        :param block: Lớp block dùng để tạo stage.
        :param planes: Số kênh đầu ra của các block trong stage.
        :param num_blocks: Số lượng block trong stage.
        :param stride: Bước trượt (stride) áp dụng cho block đầu tiên của stage.
        :return: nn.Sequential chứa các block đã được nối tiếp nhau.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = [] 
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def embedding(self, x):
        """
        Hàm trích xuất vector embedding (đặc trưng) của ảnh, qua 3 stage rồi global average pooling.
        :param x: Tensor ảnh đầu vào, kích thước (batch, 3, H, W).
        :return: Tensor embedding đã flatten, kích thước (batch, 64 * block.expansion).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out
    def forward(self, x):
        """
        Hàm lan truyền xuôi, trích xuất embedding rồi đưa qua lớp linear để ra logits.
        :param x: Tensor ảnh đầu vào, kích thước (batch, 3, H, W).
        :return: Tensor logits, kích thước (batch, num_classes).
        """
        out = self.embedding(x)
        return self.linear(out)

def resnet20(num_classes=10):
    """
    Hàm khởi tạo mô hình ResNet-20 (3 stage, mỗi stage 3 BasicBlock) cho ảnh dạng CIFAR.
    :param num_classes: Số lớp đầu ra của mô hình.
    :return: Đối tượng ResNet đã khởi tạo với cấu hình ResNet-20.
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)