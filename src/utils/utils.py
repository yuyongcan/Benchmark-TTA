import argparse
import logging
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def mean(items):
    return sum(items) / len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)

    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)
            correct += (predictions == labels.to(device)).float().sum()
            pass

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy


def split_up_model(model):
    modules = list(model.children())[:-1]
    classifier = list(model.children())[-1]
    while not isinstance(classifier, nn.Linear):
        sub_modules = list(classifier.children())[:-1]
        modules.extend(sub_modules)
        classifier = list(classifier.children())[-1]
    featurizer = nn.Sequential(*modules)

    return featurizer, classifier


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_output(encoder, classifier, x,arch):
    x = encoder(x)
    if arch=='WideResNet':
        x = F.avg_pool2d(x, 8)
        features = x.view(-1, classifier.in_features)
    elif arch=='vit':
        features = x[:, 0]
    else:
        features = x.squeeze()
    return features, classifier(features)


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cal_acc(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs=model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0])

    return accuracy * 100


def del_wn_hook(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)


def restore_wn_hook(model, name='weight'):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, name)


def deepcopy_model(model):
    ### del weight norm hook
    del_wn_hook(model)

    ### copy model
    model_cp = deepcopy(model)

    ### restore weight norm hook
    restore_wn_hook(model)
    restore_wn_hook(model_cp)

    return model_cp


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--MODEL_CONTINUAL', default=None, type=str)
    parser.add_argument('--OPTIM_LR', default=None, type=float)
    parser.add_argument('--BN_ALPHA', default=None, type=float)
    parser.add_argument('--output_dir', default=None, type=str, help='path to output_dir file')
    parser.add_argument('--COTTA_RST', default=None, type=float)
    parser.add_argument('--COTTA_AP', default=None, type=float)
    parser.add_argument('--M_TEACHER_MOMENTUM', default=None, type=float)
    parser.add_argument('--EATA_DM', default=None, type=float)
    parser.add_argument('--EATA_FISHER_ALPHA', default=None, type=float)
    parser.add_argument('--T3A_FILTER_K', default=None, type=int)
    parser.add_argument('--LAME_AFFINITY', default=None, type=str)
    parser.add_argument('--LAME_KNN', default=None, type=int)

    parser.add_argument('--TEST_EPOCH', default=None, type=int)
    parser.add_argument('--SHOT_CLS_PAR', default=None, type=float)
    parser.add_argument('--SHOT_ENT_PAR', default=None, type=float)
    parser.add_argument('--NRC_K', default=None, type=int)
    parser.add_argument('--NRC_KK', default=None, type=int)
    parser.add_argument('--SAR_RESET_CONSTANT', default=None, type=float)
    parser.add_argument('--PLUE_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_QUEUE_SIZE', default=None, type=int)
    args = parser.parse_args()
    return args


def merge_cfg_from_args(cfg, args):
    if args.MODEL_CONTINUAL is not None:
        cfg.MODEL.CONTINUAL = args.MODEL_CONTINUAL
    if args.OPTIM_LR is not None:
        cfg.OPTIM.LR = args.OPTIM_LR
    if args.BN_ALPHA is not None:
        cfg.BN.ALPHA = args.BN_ALPHA
    if args.COTTA_RST is not None:
        cfg.COTTA.RST = args.COTTA_RST
    if args.COTTA_AP is not None:
        cfg.COTTA.AP = args.COTTA_AP
    if args.M_TEACHER_MOMENTUM is not None:
        cfg.M_TEACHER.MOMENTUM = args.M_TEACHER_MOMENTUM
    if args.EATA_DM is not None:
        cfg.EATA.D_MARGIN = args.EATA_DM
    if args.EATA_FISHER_ALPHA is not None:
        cfg.EATA.FISHER_ALPHA = args.EATA_FISHER_ALPHA
    if args.T3A_FILTER_K is not None:
        cfg.T3A.FILTER_K = args.T3A_FILTER_K
    if args.LAME_AFFINITY is not None:
        cfg.LAME.AFFINITY = args.LAME_AFFINITY
    if args.LAME_KNN is not None:
        cfg.LAME.KNN = args.LAME_KNN
    if args.TEST_EPOCH is not None:
        cfg.TEST.EPOCH = args.TEST_EPOCH
    if args.SHOT_CLS_PAR is not None:
        cfg.SHOT.CLS_PAR = args.SHOT_CLS_PAR
    if args.SHOT_ENT_PAR is not None:
        cfg.SHOT.ENT_PAR = args.SHOT_ENT_PAR
    if args.NRC_K is not None:
        cfg.NRC.K = args.NRC_K
    if args.NRC_KK is not None:
        cfg.NRC.KK = args.NRC_KK
    if args.SAR_RESET_CONSTANT is not None:
        cfg.SAR.RESET_CONSTANT = args.SAR_RESET_CONSTANT
    if args.PLUE_NUM_NEIGHBORS is not None:
        cfg.PLUE.NUM_NEIGHBORS = args.PLUE_NUM_NEIGHBORS
    if args.ADACONTRAST_NUM_NEIGHBORS is not None:
        cfg.ADACONTRAST.NUM_NEIGHBORS = args.ADACONTRAST_NUM_NEIGHBORS
    if args.ADACONTRAST_QUEUE_SIZE is not None:
        cfg.ADACONTRAST.QUEUE_SIZE = args.ADACONTRAST_QUEUE_SIZE

