import logging
import os

import numpy as np
from scipy.spatial.distance import cdist
from timm.models.vision_transformer import VisionTransformer
# from torchvision.models.vision_transformer import VisionTransformer

from .setup import setup_shot_optimizer
from ..data.data import load_dataset_idx
from ..models import *
from ..models.base_model import BaseModel
from ..utils.conf import get_num_classes
from ..utils.utils import split_up_model, get_output, lr_scheduler, Entropy, cal_acc

logger = logging.getLogger(__name__)


def obtain_label(loader, model, cfg):
    start_test = True
    with torch.no_grad():
        for inputs, labels, idx in loader:
            inputs = inputs.cuda()
            feas, outputs = model(inputs, return_feats=True)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.SHOT.DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > cfg.SHOT.THRESHOLD)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], cfg.SHOT.DISTANCE)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logger.info(log_str)

    return predict.astype('int')


def train_target(cfg, domain, severity=5, type='eval'):
    dataset, loader = load_dataset_idx(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                       cfg.TEST.BATCH_SIZE,
                                       split='all', domain=domain, level=severity,
                                       adaptation=cfg.MODEL.ADAPTATION,
                                       workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                       ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                       num_aug=cfg.TEST.N_AUGMENTATIONS)
    ## set base network
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                       domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    model = model.cuda()
    model = BaseModel(model, cfg.MODEL.ARCH)
    optimizer = setup_shot_optimizer(model, cfg)
    arch=cfg.MODEL.ARCH
    # building feature bank and score bank
    if arch!='vit':
        model.fc.eval()
    max_iter = cfg.TEST.EPOCH * len(loader)
    iter_num = 0
    max_acc = 0
    best_epoch = 0
    for epoch in range(cfg.TEST.EPOCH):
        if cfg.SHOT.CLS_PAR > 0:
            if arch!='vit':
                model.encoder.eval()
            else:
                model.eval()
            mem_label = obtain_label(loader, model, cfg)
            mem_label = torch.from_numpy(mem_label).cuda()
            if arch != 'vit':
                model.encoder.train()
            else:
                model.train()
        for inputs_test, _, tar_idx in loader:
            iter_num += 1
            inputs_test = inputs_test.cuda()
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            features_test, outputs_test = model(inputs_test, return_feats=True)

            if cfg.SHOT.CLS_PAR > 0:
                pred = mem_label[tar_idx]
                classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
                classifier_loss *= cfg.SHOT.CLS_PAR
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.SHOT.EPSILION))
            entropy_loss -= gentropy_loss
            im_loss = entropy_loss * cfg.SHOT.ENT_PAR
            classifier_loss += im_loss

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()
        if type == 'val' or type == 'eval' and epoch == cfg.TEST.EPOCH - 1:
            model.eval()
            acc = cal_acc(loader, model)
            logger.info(f"Epoch: {epoch}, acc: {acc:.2f}%")
            if arch != 'vit':
                model.encoder.train()
            else:
                model.train()
            if acc > max_acc:
                max_acc = acc
                best_epoch = epoch
    if type=='val':
        logger.info(f"Best epoch: {best_epoch}, acc: {max_acc:.2f}%")
        return max_acc/100.
    else:
        return acc/100.
