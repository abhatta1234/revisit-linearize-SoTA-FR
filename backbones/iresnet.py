"""
IResNet backbone implementation for face recognition

Reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']

using_ckpt = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class IBasicBlock(nn.Module):
    """
    IResNet Basic Block with pre-activation design and PReLU activation.
    Uses BatchNorm -> Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm structure.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    """
    IResNet architecture for face recognition with attention mechanism.

    Key features:
    - Uses PReLU activation instead of ReLU
    - Applies Gaussian attention weighting before global average pooling
    - Pre-computed attention kernel for efficiency
    - BatchNorm1D for final feature normalization
    """
    fc_scale = 1 * 1

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()

        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2]
        )

        # Final layers
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Pre-computed Gaussian attention kernel (7x7 with sigma=3)
        # This applies spatial attention weighting to feature maps before pooling
        # Helps the model focus on central facial features while reducing edge artifacts
        self.precomputed_attention = self.gaussian_kernel(7, 3).cuda()

        # Initialize BatchNorm1D features with constant weight = 1.0 (non-trainable)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def gaussian_kernel(self, kernel_size, sigma):
        """
        Generate a 2D Gaussian kernel for spatial attention.

        This kernel is applied to feature maps before global average pooling
        to create a spatial attention mechanism that emphasizes central regions
        of the face while de-emphasizing peripheral areas.

        Args:
            kernel_size (int): Size of the square kernel (e.g., 7 for 7x7)
            sigma (float): Standard deviation of the Gaussian distribution

        Returns:
            torch.Tensor: Normalized Gaussian kernel with sum = 1.0
        """
        kernel_size = int(kernel_size)
        sigma = float(sigma)
        kernel = torch.zeros(kernel_size, kernel_size)
        center = torch.tensor(kernel_size // 2)

        for x in range(kernel_size):
            for y in range(kernel_size):
                # Calculate Gaussian weight based on distance from center
                kernel[x, y] = torch.exp(
                    -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
                )

        # Normalize kernel so weights sum to 1
        return kernel / kernel.sum()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Create a ResNet layer with specified number of blocks."""
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through IResNet.

        Key steps:
        1. Initial conv + BN + PReLU
        2. Four ResNet layers with increasing channels
        3. Final BatchNorm
        4. Gaussian attention weighting (spatial attention)
        5. Global average pooling
        6. Dropout + FC + feature normalization
        """
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn2(x)

            # Apply pre-computed Gaussian attention mask
            # This weights the feature maps spatially before pooling
            # The attention kernel is expanded to match feature map dimensions
            x = x * self.precomputed_attention.expand_as(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)

        # Convert to float32 for FC layer if using fp16
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)  # Final feature normalization
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    """Internal function to create IResNet models."""
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("Pretrained models not implemented")
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    """IResNet-18 model with [2, 2, 2, 2] layer configuration."""
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    """IResNet-34 model with [3, 4, 6, 3] layer configuration."""
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    """IResNet-50 model with [3, 4, 14, 3] layer configuration."""
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    """IResNet-100 model with [3, 13, 30, 3] layer configuration."""
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    """IResNet-200 model with [6, 26, 60, 6] layer configuration."""
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs)