import torch

from .CifarResNeXt import CifarResNeXt, ResNeXtBottleneck


class Hendrycks2020AugMixResNeXtNet(CifarResNeXt):
    def __init__(self, depth=29, cardinality=4, base_width=32):
        super().__init__(ResNeXtBottleneck,
                         depth=depth,
                         num_classes=100,
                         cardinality=cardinality,
                         base_width=base_width)
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)
