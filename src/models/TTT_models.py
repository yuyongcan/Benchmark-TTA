import torch.nn as nn
from ..utils.utils import split_up_model
from .WideResNet import WideResNet
import torch.nn.functional as F
class TTT_base(nn.Module):
    def __init__(self, model, mode='train'):
        super().__init__()
        self.model = model
        self.mode = mode
        self.featurizer, self.classifier = split_up_model(model)
        self.ssHead = nn.Linear(self.classifier.in_features, 4)

    def forward(self, x):
        features = self.featurizer(x)
        if isinstance(self.model, WideResNet):
            features = F.avg_pool2d(features, 8)
            features = features.view(-1, self.model.nChannels)
        logits = self.classifier(features)
        rotations = self.ssHead(features)
        return logits, rotations
