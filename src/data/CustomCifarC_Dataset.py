import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
class CustomCifarC_Dataset(torch.utils.data.TensorDataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        super().__init__(tensors[0], tensors[1])
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = transforms.ToPILImage()(x)
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)