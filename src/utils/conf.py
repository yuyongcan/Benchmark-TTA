# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information

# Data directory
_C.DATA_DIR = "/data2/yongcan.yu/datasets"

# Weight directory
_C.CKPT_DIR = "./ckpt/"

# Output directory
_C.OUTPUT = "./output"

# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 2020

# Deterministic experiments.
_C.DETERMINISM = False

# The num_workers argument to use in the data loaders
_C.WORKERS = 4

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench or https://pytorch.org/vision/0.14/models.html for available models
_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights for torchvision models. See: https://pytorch.org/vision/0.14/models.html
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

_C.MODEL.CONTINUAL = 'Fully'

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10_c'

_C.CORRUPTION.SOURCE_DATASET = 'cifar10'

_C.CORRUPTION.SOURCE_DOMAIN = 'origin'
_C.CORRUPTION.SOURCE_DOMAINS = ['origin']
# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate (10000 for all samples in CIFAR-C)
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
# If num_ex is larger than 5000 all images (50,000) are loaded and then subsampled to num_ex
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# ------------------------------------- T3A options ------------------------- #
_C.T3A = CfgNode()
_C.T3A.FILTER_K = 10

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1 / 3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 1.0
_C.RMT.LAMBDA_CE_TRG = 1.0
_C.RMT.LAMBDA_CONT = 1.0
_C.RMT.NUM_SAMPLES_WARM_UP = 50000

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard"  # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0  # lambda for classification loss
_C.ADACONTRAST.BETA = 1.0  # lambda for instance loss
_C.ADACONTRAST.ETA = 1.0  # lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine"  # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong"  # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LAME options ----------------------------- #
_C.LAME = CfgNode()

_C.LAME.AFFINITY = "rbf"
_C.LAME.KNN = 5
_C.LAME.SIGMA = 1.0
_C.LAME.FORCE_SYMMETRY = False

# --------------------------------- SAR options ----------------------------- #
_C.SAR = CfgNode()
_C.SAR.RESET_CONSTANT = 0.2
_C.SAR.E_MARGIN_COE = 0.4

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha
_C.EATA.FISHER_ALPHA = 2000

# Number of samples for ewc regularization
_C.EATA.NUM_SAMPLES = 2000

# Diversity margin
_C.EATA.D_MARGIN = 0.05

_C.EATA.E_MARGIN_COE = 0.4

# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0  # [0, 1] possibility to reduce the number of source samples

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 64

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

_C.TEST.EPOCH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# ------------------------------- NRC options -------------------------- #
_C.NRC = CfgNode()
_C.NRC.K = 5
_C.NRC.KK = 5
_C.NRC.EPSILION = 1e-5

# ------------------------------- SHOT options -------------------------- #
_C.SHOT = CfgNode()
_C.SHOT.EPSILION = 1e-5
_C.SHOT.CLS_PAR = 0.3
_C.SHOT.DISTANCE = 'cosine'
_C.SHOT.THRESHOLD = 0
_C.SHOT.ENT_PAR = 1

# ------------------------------- PLUE options -------------------------- #
_C.PLUE = CfgNode()
_C.PLUE.CTR= True
_C.PLUE.NUM_NEIGHBORS = 10
_C.PLUE.TEMPORAL_LENGTH = 5
_C.PLUE.TEMPERATURE = 0.07
_C.PLUE.LABEL_REFINEMENT = True
_C.PLUE.NEG_L = True
_C.PLUE.REWEIGHTING = True

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(cfg_file, output_dir):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    merge_from_file(cfg_file)

    log_dest = os.path.basename(cfg_file)
    t = time.time()
    t = int(t * 1000000) % 100000
    log_dest = log_dest.replace('.yaml', '{}_{}.txt'.format(current_time, str(t)))

    cfg.OUTPUT = os.path.join(cfg.OUTPUT, output_dir)
    g_pathmgr.mkdirs(cfg.OUTPUT)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    # cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.OUTPUT, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",  # do not change
               "imagenet_d109": "imagenet-d",  # do not change
               "domainnet126": "DomainNet-126",
               # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               }
    return os.path.join(root, mapping[dataset_name])


def get_num_classes(dataset_name):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100, "cifar100_c": 100,
                                "imagenet": 1000, "imagenet_c": 1000, "imagenet_k": 1000, "imagenet_r": 200,
                                "imagenet_a": 200, "imagenet_d": 164, "imagenet_d109": 109, "imagenet200": 200,
                                "domainnet126": 126, "office31": 31, "visda": 12, "officehome": 65
                                }
    return dataset_name2num_classes[dataset_name]


def get_domain_sequence(dataset, domain):
    mapping = {}
    mapping['domainnet126'] = {"real": ["clipart", "painting", "sketch"],
                               "clipart": ["painting", "real", "sketch"],
                               "painting": ["clipart", "real", "sketch"],
                               "sketch": ["clipart", "painting", "real"],
                               }
    mapping['officehome'] = {'Art': ['Clipart', 'Product', 'RealWorld'],
                             'Clipart': ['Art', 'Product', 'RealWorld'],
                             'Product': ['Art', 'Clipart', 'RealWorld'],
                             'RealWorld': ['Art', 'Clipart', 'Product']}
    return mapping[dataset][domain]
