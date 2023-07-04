import argparse
import logging
import time

import numpy as np

import src.methods.nrc as NRC
import src.methods.shot as SHOT
import src.methods.plue as PLUE
from src.utils.conf import cfg, load_cfg_fom_args, get_domain_sequence
from src.utils.utils import merge_cfg_from_args, get_args

logger = logging.getLogger(__name__)


def validation(cfg):
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
    dom_names_loop = [dom_names_loop[0]]
    accs = []

    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if cfg.MODEL.ADAPTATION == "nrc":
            acc = NRC.train_target(cfg, domain_name, severity=5, type='val')
        if cfg.MODEL.ADAPTATION == "shot":
            acc = SHOT.train_target(cfg, domain_name, severity=5, type='val')
        if cfg.MODEL.ADAPTATION == "plue":
            acc = PLUE.train_target(cfg, domain_name, severity=5, type='val')
        accs.append(acc)
        logger.info(f"domain {domain_name} accuracy: {acc:.2%}")
    return np.mean(accs)




if __name__ == "__main__":
    args = get_args()
    args.output_dir = args.output_dir if args.output_dir else 'SFDA_validation'
    load_cfg_fom_args(args.cfg, args.output_dir)
    merge_cfg_from_args(cfg, args)
    logger.info(cfg)
    start_time = time.time()
    accs = []
    for domain in [cfg.CORRUPTION.SOURCE_DOMAINS[0]]:
        logger.info("#" * 50 + f'evaluating domain {domain}' + "#" * 50)
        cfg.CORRUPTION.SOURCE_DOMAIN = domain
        acc = validation(cfg)
        accs.append(acc)

    logger.info("#" * 50 + 'fianl result' + "#" * 50)
    logger.info(f"total mean accuracy: {np.mean(accs):.2%}")

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
