"""
---file model_resnet.py---

Definition of the ResNet-20 model (CIFAR-style ResNet).

Implements a CIFAR-oriented ResNet built from basic residual blocks: an initial
3x3 convolution followed by 3 stages of BasicBlocks, global average pooling, and
a final linear classifier. Provides the `resnet20` factory function and an
`embedding` method for extracting feature vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    """
    Initialize layer weights: Kaiming normal for Conv2d/Linear layers, and
    (weight=1, bias=0) for BatchNorm2d layers.

    :param m: The module (layer) of the model whose weights are initialized.
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
    Basic residual block of ResNet, consisting of two 3x3 conv layers with
    BatchNorm and a shortcut (skip) connection.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """
        Initialize a BasicBlock, automatically creating a shortcut projection
        if the number of channels or the stride changes.

        :param in_planes: Number of input channels of the block.
        :param planes: Number of output channels of the block.
        :param stride: Stride of the first conv layer.
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
        Forward pass through the BasicBlock: add the shortcut connection, then
        apply ReLU.

        :param x: Input tensor, of shape (batch, in_planes, H, W).
        :return: Output tensor after the residual block, of shape
            (batch, planes, H', W').
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    CIFAR-style ResNet model (3 stages, each stage containing several
    BasicBlocks) for 32x32 images.
    """

    def __init__(self, block, num_blocks, num_classes=10):
        """
        Initialize the ResNet model: build 3 stages from the basic block, then
        initialize the weights.

        :param block: Block class used to build the stages (e.g. BasicBlock).
        :param num_blocks: List with the number of blocks in each stage
            (3 stages).
        :param num_classes: Number of output classes of the model.
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
        Build one stage consisting of several consecutive blocks; only the
        first block uses the given stride.

        :param block: Block class used to build the stage.
        :param planes: Number of output channels of the blocks in the stage.
        :param num_blocks: Number of blocks in the stage.
        :param stride: Stride applied to the first block of the stage.
        :return: An nn.Sequential containing the concatenated blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def embedding(self, x):
        """
        Extract the embedding (feature) vector of an image, passing it through
        the 3 stages followed by global average pooling.

        :param x: Input image tensor, of shape (batch, 3, H, W).
        :return: Flattened embedding tensor, of shape
            (batch, 64 * block.expansion).
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
        Forward pass: extract the embedding, then pass it through the linear
        layer to produce logits.

        :param x: Input image tensor, of shape (batch, 3, H, W).
        :return: Logits tensor, of shape (batch, num_classes).
        """
        out = self.embedding(x)
        return self.linear(out)


def resnet20(num_classes=10):
    """
    Build a ResNet-20 model (3 stages, 3 BasicBlocks each) for CIFAR-style
    images.

    :param num_classes: Number of output classes of the model.
    :return: An initialized ResNet object configured as ResNet-20.
    """
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)