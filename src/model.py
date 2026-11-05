"""Three-resolution residual CNN for image classification.

224 input -> 56/28/14 grids -> align at 56 -> concatenate and fuse -> GAP + GeM -> 12 classes.
This module defines the network; data processing and training are separate.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv_bn(in_channels, out_channels, kernel=3, stride=1, relu=True):
    """Convolution and batch normalization, with an optional ReLU activation."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    """Learn a residual update: output = ReLU(x + two-convolution branch)."""
    def __init__(self, channels):
        super().__init__()
        # The second convolution leaves the residual signed before addition.
        self.first = conv_bn(channels, channels)
        self.second = conv_bn(channels, channels, relu=False)

    def forward(self, x):
        return F.relu(x + self.second(self.first(x)))


class ExchangeModule(nn.Module):
    """Update each branch, then fuse all branches from the same set of updated features."""
    def __init__(self, channels):
        super().__init__()
        # Each resolution is updated by two residual blocks before exchange.
        self.branches = nn.ModuleList([
            nn.Sequential(BasicBlock(c), BasicBlock(c)) for c in channels
        ])
        self.paths = nn.ModuleList()
        for target_index, target_channels in enumerate(channels):
            paths_to_target = nn.ModuleList()
            for source_index, source_channels in enumerate(channels):
                if source_index == target_index:
                    path = nn.Identity()
                elif source_index > target_index:
                    # Coarse to fine: match channels with 1x1 convolution, then upsample in forward.
                    path = conv_bn(source_channels, target_channels, kernel=1, relu=False)
                else:
                    # Fine to coarse: each stride-2 convolution halves height and width.
                    layers = []
                    for step in range(target_index - source_index):
                        last = step == target_index - source_index - 1
                        output_channels = target_channels if last else source_channels
                        layers.append(conv_bn(source_channels, output_channels, stride=2, relu=not last))
                    path = nn.Sequential(*layers)
                paths_to_target.append(path)
            self.paths.append(paths_to_target)

    def forward(self, inputs):
        # Read all source features from the same update step. This avoids
        # feeding an already-exchanged branch into another target.
        sources = [branch(x) for branch, x in zip(self.branches, inputs)]
        outputs = []
        for target_index, paths in enumerate(self.paths):
            terms = []
            for source_index, path in enumerate(paths):
                feature = path(sources[source_index])
                if source_index > target_index:
                    feature = F.interpolate(
                        feature, size=sources[target_index].shape[-2:],
                        mode='bilinear', align_corners=False,
                    )
                # Every term now matches this target's channel count and grid.
                terms.append(feature)
            outputs.append(F.relu(torch.stack(terms).sum(0)))
        return outputs


class MultiResolutionCNN(nn.Module):
    """Keep one feature extractor and set the classifier width from the dataset."""
    def __init__(self, num_classes=12, seed=42):
        super().__init__()
        self.num_classes = num_classes
        # Keep initialization reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            # Two stride-2 layers turn a 224x224 image into a 56x56 feature map.
            self.stem = nn.Sequential(conv_bn(3, 32, stride=2), conv_bn(32, 48, stride=2))
            self.stage1 = nn.Sequential(BasicBlock(48), BasicBlock(48))
            # Introduce the second resolution and exchange features twice.
            self.transition2 = conv_bn(48, 96, stride=2)
            self.stage2 = nn.ModuleList([ExchangeModule([48, 96]) for _ in range(2)])
            # Introduce the third resolution, then update all three together.
            self.transition3 = conv_bn(96, 192, stride=2)
            self.stage3 = nn.ModuleList([ExchangeModule([48, 96, 192]) for _ in range(2)])
            # Project each branch to 128 channels before spatial alignment.
            self.projections = nn.ModuleList([conv_bn(c, 128, kernel=1) for c in [48, 96, 192]])
            self.fusion = conv_bn(384, 256, kernel=1)
            # Two pooled 256-channel vectors form the 512-dimensional head.
            self.norm = nn.LayerNorm(512)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(512, num_classes)
            self.apply(self.initialize)

    @staticmethod
    def initialize(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def encode(self, x):
        x = self.stage1(self.stem(x))                 # [B, 48, 56, 56]
        branches = [x, self.transition2(x)]          # Add a [B, 96, 28, 28] branch.
        for exchange in self.stage2:
            branches = exchange(branches)
        branches.append(self.transition3(branches[-1]))  # Add a [B, 192, 14, 14] branch.
        for exchange in self.stage3:
            branches = exchange(branches)

        projected = []
        for index, (projection, branch) in enumerate(zip(self.projections, branches)):
            feature = projection(branch)
            if index > 0:
                feature = F.interpolate(feature, size=branches[0].shape[-2:], mode='bilinear', align_corners=False)
            projected.append(feature)
        return self.fusion(torch.cat(projected, dim=1))  # [B, 256, 56, 56]

    def forward(self, x):
        feature = self.encode(x)
        # GAP averages responses; GeM emphasizes stronger ones with a fixed p=3.
        # Compute powers in FP32. Global pooling supports both 160 and 224 inputs.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            feature = feature.float()
            gap = feature.mean((2, 3))
            gem = feature.clamp_min(1e-6).pow(3).mean((2, 3)).pow(1 / 3)
            pooled = self.norm(torch.cat([gap, gem], dim=1))  # [B, 512]
        return self.classifier(self.dropout(pooled))       # [B, num_classes] logits, before softmax.
