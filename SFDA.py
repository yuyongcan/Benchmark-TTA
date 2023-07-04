import argparse
import logging
import time

import numpy as np

import src.methods.nrc as NRC
import src.methods.shot as SHOT
import src.methods.plue as PLUE
from src.utils.conf import cfg, load_cfg_fom_args, get_domain_sequence

logger = logging.getLogger(__name__)


def evaluate(cfg):
    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    dom_names_loop = dom_names_all

    # setup the severities for the gradual setting

    # severities = cfg.CORRUPTION.SEVERITY

    accs = []

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if cfg.MODEL.ADAPTATION == "nrc":
            acc = NRC.train_target(cfg, domain_name, severity=5, type='eval')
        if cfg.MODEL.ADAPTATION == "shot":
            acc = SHOT.train_target(cfg, domain_name, severity=5, type='eval')
        if cfg.MODEL.ADAPTATION == "plue":
            acc = PLUE.train_target(cfg, domain_name, severity=5, type='eval')
        accs.append(acc)
        logger.info(f"domain {domain_name} accuracy: {acc:.2%}")
    return np.mean(accs)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--OPTIM_LR', default=None, type=str)
    parser.add_argument('--BN_ALPHA', default=None, type=str)
    parser.add_argument('--output_dir', default='SFDA_evaluation', type=str, help='path to output_dir file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    load_cfg_fom_args(args.cfg, args.output_dir)
    logger.info(cfg)
    start_time = time.time()
    accs = []
    for domain in cfg.CORRUPTION.SOURCE_DOMAINS:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        acc = evaluate(cfg)
        accs.append(acc)

    logger.info("#" * 50 + 'fianl result' + "#" * 50)
    logger.info(f"total mean accuracy: {np.mean(accs):.2%}")

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
