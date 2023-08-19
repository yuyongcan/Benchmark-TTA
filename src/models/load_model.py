import os

import timm
from torchvision.models import resnet50, ResNet50_Weights, convnext_base, ConvNeXt_Base_Weights, efficientnet_b0, EfficientNet_B0_Weights

from ..models import *

def load_model(model_name, checkpoint_dir=None, domain=None):
    if model_name == 'Hendrycks2020AugMix_ResNeXt':
        model = Hendrycks2020AugMixResNeXtNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'Hendrycks2020AugMix_ResNeXt.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')
    elif model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'WideResNet':
        model = WideResNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'WideResNet.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')
    # elif 'domainnet126' == model_name:
    #     # domain = model_name.split('_')[-1]
    #     feature_extractor = models.domainnet126G(domain=domain, pretrained=True)
    #     classifier = models.domainnet126C(domain=domain, pretrained=True)
    #     model = torch.nn.Sequential(feature_extractor, classifier)
    # elif 'officehome' == model_name:
    #     feature_extractor = models.officehomeG(pretrained=True)
    #     classifier = models.officehomeC(domain=domain, pretrained=True)
    #     model = torch.nn.Sequential(feature_extractor, classifier)
    elif model_name == 'officehome_shot':
        model= OfficeHome_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir,'officehome',domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'domainnet126_shot':
        model= DomainNet126_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir,'domainnet126',domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'vit':
        model=timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'convnext_base':
        model=convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        raise ValueError('Unknown model name')

    return model
