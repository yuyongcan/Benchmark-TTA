import device as device
import torch.nn as nn
from ..models import network
class OfficeHome_Shot(nn.Module):
    def __init__(self):
        super(OfficeHome_Shot, self).__init__()
        self.netF = network.ResBase(res_name='resnet50')
        self.netB = network.feat_bottleneck(type='bn', feature_dim=self.netF.in_features,
                                       bottleneck_dim=256)
        self.netC = network.feat_classifier(type='wn', class_num=65, bottleneck_dim=256)

        pass

    def forward(self, x):
        x = self.netF(x)
        x = self.netB(x)
        x = self.netC(x)
        return x
