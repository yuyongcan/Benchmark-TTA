import os
import pickle

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from robustbench.data import load_cifar10c, load_cifar100c
from .CustomCifarC_Dataset import CustomCifarC_Dataset
from .Dataset_Idx import Dataset_Idx
from .DomainNet126 import DomainNet126
from .augmentations import get_augmentation_versions, NCropsTransform
from .augmentations.transforms_memo_cifar import aug_cifar
from .augmentations.transforms_memo_imagenet import aug_imagenet
from .data_list import *
from .selectedRotateImageFolder import SelectedRotateImageFolder

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    transforms.ToTensor(),
                                    normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])


def get_transform(dataset_name, adaptation, num_augment=1):
    """
    Get transformation pipeline
    Note that the data normalization is done inside of the model
    :param dataset_name: Name of the dataset
    :param adaptation: Name of the adaptation method
    :return: transforms
    """
    if adaptation in {"adacontrast", "plue"}:
        # adacontrast requires specific transformations
        if dataset_name in {"cifar10", "cifar100", "cifar10_c", "cifar100_c"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=32,
                                                  crop_size=32)
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized and centre cropped
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=224,
                                                  crop_size=224)
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224)
        else:
            # use classical ImageNet transformation procedure
            transform = get_augmentation_versions(aug_versions="iwss", aug_type="moco-v2", res_size=256, crop_size=224)
    elif adaptation == "memo":
        original_transform = get_transform(dataset_name, None)
        transform_list = [original_transform]
        if 'cifar' in dataset_name:
            transform_aug = aug_cifar
            transforms_one = transforms.Compose([original_transform, transform_aug])
        else:
            transforms_list = original_transform.transforms[:-1]
            transforms_list.append(aug_imagenet)
            transforms_list.append(normalize)
            transforms_one = transforms.Compose(transforms_list)

        for i in range(num_augment):
            transform_list.append(transforms_one)
        transform = NCropsTransform(transform_list)

    else:
        # create non-method specific transformation
        if 'cifar' in dataset_name:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name == "imagenet_c":
            # note that ImageNet-C is already resized an centre cropped
            transform = transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize])
        else:
            # use classical ImageNet transformation procedure
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

    return transform


def load_imagenet_c(root, corruption, transforms, level=5, batch_size=64, workers=4, ckpt=None):
    assert os.path.exists(root), f'Path {root} does not exist'
    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                          'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                          'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                          'zoom_blur'], f'Unknown corruption: {corruption}'

    validdir = os.path.join(root, corruption, str(level))
    teset = SelectedRotateImageFolder(validdir, transforms, original=False,
                                      rotation=False)

    ckpt_dir = os.path.join(ckpt, 'imagenet_c')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'list.pickle')
    if not os.path.exists(ckpt_path):
        idx = torch.randperm(len(teset))
        idx = [i.item() for i in idx]
        with open(ckpt_path, 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open(ckpt_path, 'rb') as f:
            idx = pickle.load(f)
    teset.samples = [teset.samples[i] for i in idx]
    teset.switch_mode(True, False)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False,
                                           num_workers=workers, pin_memory=True)

    return teset, teloader


def load_cifar10_c(root, corruption, level=5, batch_size=64, workers=4, transforms=None, ckpt=None):
    assert os.path.exists(root), f'Path {root} does not exist'
    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                          'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                          'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                          'zoom_blur'], f'Unknown corruption: {corruption}'
    xtest, ytest = load_cifar10c(n_examples=10000, severity=level, data_dir=root, shuffle=False,
                                 corruptions=[corruption])
    teset = CustomCifarC_Dataset((xtest, ytest), transform=transforms)

    ckpt_dir = os.path.join(ckpt, 'cifar10_c')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'list.pickle')
    if not os.path.exists(ckpt_path):
        idx = torch.randperm(len(teset))
        idx = [i.item() for i in idx]
        with open(ckpt_path, 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open(ckpt_path, 'rb') as f:
            idx = pickle.load(f)
    teset = torch.utils.data.Subset(teset, idx)

    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                           pin_memory=True)
    return teset, teloader


def load_cifar100_c(root, corruption, level=5, batch_size=64, workers=4, transforms=None, ckpt=None):
    assert os.path.exists(root), f'Path {root} does not exist'
    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                          'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                          'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                          'zoom_blur'], f'Unknown corruption: {corruption}'

    xtest, ytest = load_cifar100c(n_examples=10000, severity=level, data_dir=root, shuffle=False,
                                  corruptions=[corruption])
    teset = CustomCifarC_Dataset((xtest, ytest), transform=transforms)

    ckpt_dir = os.path.join(ckpt, 'cifar100_c')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'list.pickle')
    if not os.path.exists(ckpt_path):
        idx = torch.randperm(len(teset))
        idx = [i.item() for i in idx]
        with open(ckpt_path, 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open(ckpt_path, 'rb') as f:
            idx = pickle.load(f)
    teset = torch.utils.data.Subset(teset, idx)

    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                           pin_memory=True)
    return teset, teloader


def load_cifar10(root, batch_size=64, workers=4, split="train", transforms=None):
    assert os.path.exists(root), 'CIFAR10 root path does not exist: {}'.format(root)
    if split == 'train':
        dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                               transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
    elif split == 'val':
        dataset = torchvision.datasets.CIFAR10(root=root, train=False,
                                               transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
    elif split == 'all':
        dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                               transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
        dataset2 = torchvision.datasets.CIFAR10(root=root, train=False,
                                                transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
        dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=True)

    return dataset, data_loader


def load_cifar100(root, batch_size=64, workers=4, split="train", transforms=None):
    assert os.path.exists(root), 'CIFAR100 root path does not exist: {}'.format(root)
    if split == 'train':
        dataset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
    elif split == 'val':
        dataset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
    elif split == 'all':
        dataset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
        dataset2 = torchvision.datasets.CIFAR100(root=root, train=False,
                                                 transform=torchvision.transforms.ToTensor() if transforms is None else transforms)
        dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
    else:
        raise ValueError(f'Unknown split: {split}')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=True)
    return dataset, data_loader


def load_imagenet(root, batch_size=64, workers=1, split="val", transforms=None, ckpt=None):
    assert os.path.exists(root), 'ImageNet root path does not exist: {}'.format(root)
    dataset = torchvision.datasets.ImageNet(root=os.path.join(root, 'ImageNet'), split=split,
                                            transform=te_transforms if transforms is None else transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=workers, pin_memory=True)

    return dataset, data_loader


def load_domainnet126(root, domain, transforms, batch_size=64, workers=4, split='train'):
    assert os.path.exists(root), 'DomainNet root path does not exist: {}'.format(root)
    assert domain in ['clipart', 'painting', 'real', 'sketch'], f'Unknown domain: {domain}'

    if split == 'train':
        dataset = DomainNet126(root=root, transform=transforms, domain=domain, train=True, download=True)
    elif split == 'val':
        dataset = DomainNet126(root=root, transform=transforms, domain=domain, train=False, download=True)
    elif split == 'all':
        train_dataset = DomainNet126(root=root, transform=transforms, domain=domain, train=True,
                                     download=True)
        val_dataset = DomainNet126(root=root, transform=transforms, domain=domain, train=False,
                                   download=True)
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                              pin_memory=True)
    return dataset, data_loader


def load_officehome(root, domain, transforms=None, batch_size=64, workers=4, split='train'):
    data_dir = os.path.join(root, 'office-home')

    txt_train_path = os.path.join(data_dir, domain + "_train.pickle")
    txt_test_path = os.path.join(data_dir, domain + "_test.pickle")

    txt_src = open(os.path.join(data_dir, domain + '_list.txt')).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    if not os.path.exists(txt_train_path) or not os.path.exists(txt_test_path):
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        with open(txt_train_path, 'wb') as f:
            pickle.dump(tr_txt, f)
        with open(txt_test_path, 'wb') as f:
            pickle.dump(te_txt, f)
    else:
        with open(txt_train_path, 'rb') as f:
            tr_txt = pickle.load(f)
        with open(txt_test_path, 'rb') as f:
            te_txt = pickle.load(f)

    if split == "train":
        dataset = ImageList(tr_txt, transform=image_train() if transforms is None else transforms)
    elif split == "val":
        dataset = ImageList(te_txt, transform=image_test() if transforms is None else transforms)
    elif split == "all":
        all_txt = tr_txt + te_txt
        dataset = ImageList(all_txt, transform=image_test() if transforms is None else transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
    return dataset, dataloader


def load_dataset(dataset, root, batch_size=64, workers=4, split='train', adaptation=None, domain=None,
                 level=None, ckpt=None, num_aug=1, transforms=None):
    transforms = get_transform(dataset, adaptation, num_aug) if transforms is None else transforms
    if dataset == 'cifar10':
        return load_cifar10(root=root, batch_size=batch_size, workers=workers, split=split, transforms=transforms)
    elif dataset == 'cifar100':
        return load_cifar100(root=root, batch_size=batch_size, workers=workers, split=split, transforms=transforms)
    elif dataset == 'imagenet':
        return load_imagenet(root=root, batch_size=batch_size, workers=workers, split=split, transforms=transforms,
                             ckpt=ckpt)
    elif dataset == 'domainnet126':
        return load_domainnet126(root=root, domain=domain, batch_size=batch_size, workers=workers, split=split,
                                 transforms=transforms)
    elif dataset == 'cifar10_c':
        return load_cifar10_c(root=root, corruption=domain, level=level, batch_size=batch_size, workers=workers,
                              transforms=transforms, ckpt=ckpt)
    elif dataset == 'cifar100_c':
        return load_cifar100_c(root=root, corruption=domain, level=level, batch_size=batch_size, workers=workers,
                               transforms=transforms, ckpt=ckpt)
    elif dataset == 'imagenet_c':
        return load_imagenet_c(root=os.path.join(root, 'ImageNet-C'), batch_size=batch_size, corruption=domain,
                               level=level, workers=workers,
                               transforms=transforms, ckpt=ckpt)
    elif dataset == 'officehome':
        return load_officehome(root=root, domain=domain, batch_size=batch_size, workers=workers, split=split,
                               transforms=transforms)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def load_dataset_idx(dataset, root, batch_size=64, workers=4, split='train', adaptation=None, domain=None,
                     level=None, ckpt=None, num_aug=1):
    dataset, _ = load_dataset(dataset, root, batch_size, workers, split, adaptation, domain, level, ckpt, num_aug)
    dataset_idx = Dataset_Idx(dataset)
    data_loader = torch.utils.data.DataLoader(dataset_idx, batch_size=batch_size, shuffle=False, num_workers=workers,
                                              drop_last=False)
    return dataset_idx, data_loader
