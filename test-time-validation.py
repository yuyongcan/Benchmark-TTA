import logging
import os

import numpy as np

from src.methods import *
from src.models.load_model import load_model
from src.utils import get_accuracy, merge_cfg_from_args, get_args
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence

logger = logging.getLogger(__name__)


def validation(cfg):
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    base_model = base_model.cuda()

    logger.info(f"Setting up test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")
    if cfg.MODEL.ADAPTATION == "source":  # BN--0
        model, param_names = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "t3a":
        model, param_names = setup_t3a(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
        model, param_names = setup_test_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
        model, param_names = setup_alpha_norm(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "memo":
        model, param_names = setup_memo(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "tent":
        model, param_names = setup_tent(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "cotta":
        model, param_names = setup_cotta(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "lame":
        model, param_names = setup_lame(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "adacontrast":
        model, param_names = setup_adacontrast(base_model, cfg)
    elif cfg.MODEL.ADAPTATION == "eata":
        model, param_names = setup_eata(base_model, num_classes, cfg)
    elif cfg.MODEL.ADAPTATION == "sar":
        model = setup_sar(base_model, cfg, num_classes)
    else:
        raise ValueError(f"Adaptation method '{cfg.MODEL.ADAPTATION}' is not supported!")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    if cfg.MODEL.CONTINUAL=='Fully':
        dom_names_loop = [dom_names_all[0]]
    elif cfg.MODEL.CONTINUAL=='Continual':
        dom_names_loop = dom_names_all
    # setup the severities for the gradual setting

    severities = [cfg.CORRUPTION.SEVERITY[0]]

    accs = []
    domain_dict = {}

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if cfg.MODEL.CONTINUAL == 'Fully':
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        elif cfg.MODEL.CONTINUAL == 'Continual':
            logger.warning("not resetting model")

        for severity in severities:
            testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                                cfg.TEST.BATCH_SIZE,
                                                split='all', domain=domain_name, level=severity,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                num_aug=cfg.TEST.N_AUGMENTATIONS)

            for epoch in range(cfg.TEST.EPOCH):
                acc = get_accuracy(
                    model, data_loader=test_loader)
                print(f"epoch: {epoch}, acc: {acc:.2%}")
            accs.append(acc)

            logger.info(
                f"{cfg.CORRUPTION.DATASET} accuracy % [{domain_name}{severity}][#samples={len(testset)}]: {acc:.2%}")

        logger.info(f"mean accuracy: {np.mean(accs):.2%}")


if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'validation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    merge_cfg_from_args(cfg, args)
    cfg.CORRUPTION.SOURCE_DOMAIN = cfg.CORRUPTION.SOURCE_DOMAINS[0]
    logger.info(cfg)
    validation(cfg)
