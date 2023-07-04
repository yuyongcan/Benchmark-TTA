import logging
import os

from tqdm import tqdm

from .setup import setup_NRC_optimizer
from ..data.data import load_dataset_idx
from ..models import *
from ..models.base_model import BaseModel
from ..utils import AverageMeter
from ..utils.conf import get_num_classes
from ..utils.utils import split_up_model, lr_scheduler, get_output

logger = logging.getLogger(__name__)


def train_target(cfg, domain, severity=5, type='eval'):
    dataset, loader = load_dataset_idx(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                       cfg.TEST.BATCH_SIZE,
                                       split='all', domain=domain, level=severity,
                                       adaptation=cfg.MODEL.ADAPTATION,
                                       workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                       ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                       num_aug=cfg.TEST.N_AUGMENTATIONS)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True,
                                         num_workers=cfg.TEST.NUM_WORKERS, pin_memory=True)
    ## set base network
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    model = load_model(model_name=cfg.MODEL.ARCH, checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                       domain=cfg.CORRUPTION.SOURCE_DOMAIN)
    model = model.cuda()
    optimizer, optimizer_c = setup_NRC_optimizer(model, cfg)
    arch=cfg.MODEL.ARCH
    model = BaseModel(model, arch)
    feature_size = model._output_dim
    # building feature bank and score bank
    num_sample = len(dataset)
    fea_bank = torch.randn(num_sample, feature_size)
    score_bank = torch.randn(num_sample, num_classes).cuda()

    model.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        acces_eval = AverageMeter(name='acc')
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            labels = data[1]
            inputs = inputs.cuda()
            output, outputs = model(inputs,return_feats=True)
            output_norm = F.normalize(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()
            acc = (outputs.argmax(dim=1) == labels.cuda()).float().mean()
            acces_eval.update(acc.item(), inputs.size(0))
        logger.info(f"originl acc: {acces_eval.avg*100:.2f}%")

    max_iter = cfg.TEST.EPOCH * len(loader)
    iter_num = 0

    model.train()
    eval_accs = []
    best_epoch = 0
    for epoch in range(cfg.TEST.EPOCH):
        losses = AverageMeter(name='loss')
        acces = AverageMeter(name='acc')
        loop = tqdm((loader), leave=True, total=len(loader))
        for inputs_test, labels, tar_idx in loop:
            inputs_test = inputs_test.cuda()
            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
            lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

            features_test, outputs_test = model(inputs_test, return_feats=True)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            # output_re = softmax_out.unsqueeze(1)
            acc = (softmax_out.argmax(dim=1) == labels.cuda()).float().mean()

            with torch.no_grad():
                output_f_norm = F.normalize(features_test)
                output_f_ = output_f_norm.cpu().detach().clone()

                fea_bank[tar_idx] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = softmax_out.detach().clone()

                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance,
                                         dim=-1,
                                         largest=True,
                                         k=cfg.NRC.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C

                fea_near = fea_bank[idx_near]  # batch x K x num_dim
                fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                              k=cfg.NRC.KK + 1)  # M near neighbors for each of above K ones
                idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
                match = (
                        idx_near_near == tar_idx_).sum(-1).float()  # batch x K
                weight = torch.where(
                    match > 0., match,
                    torch.ones_like(match).fill_(0.1))  # batch x K

                weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                        cfg.NRC.KK)  # batch x K x M
                weight_kk = weight_kk.fill_(0.1)

                # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                # weight_kk[idx_near_near == tar_idx_]=0

                score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                # print(weight_kk.shape)
                weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                        -1)  # batch x KM

                score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                num_classes)  # batch x KM x C

            # nn of nn
            output_re = softmax_out.unsqueeze(1).expand(-1, cfg.NRC.K * cfg.NRC.KK,
                                                        -1)  # batch x C x 1
            const = torch.mean(
                (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                 weight_kk.cuda()).sum(
                    1))  # kl_div here equals to dot product since we do not use log for score_near_kk
            loss = torch.mean(const)

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, cfg.NRC.K,
                                                             -1)  # batch x K x C

            loss += torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
                                weight.cuda()).sum(1))

            # self, if not explicitly removing the self feature in expanded neighbor then no need for this
            # loss += -torch.mean((softmax_out * score_self).sum(-1))

            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(msoftmax *
                                      torch.log(msoftmax + cfg.NRC.EPSILION))
            loss += gentropy_loss

            losses.update(loss.item(), inputs_test.size(0))
            acces.update(acc.item(), inputs_test.size(0))
            loop.set_description(f'Epoch [{epoch}/{cfg.TEST.EPOCH}]')
            loop.set_postfix(loss=losses.avg, acc=f"{acces.avg * 100:.2f}%")
            optimizer.zero_grad()
            optimizer_c.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_c.step()
        if type=='val' or epoch==cfg.TEST.EPOCH-1:
            model.eval()
            acces_eval = AverageMeter(name='acc')
            for inputs, labels, idx in loader:
                inputs = inputs.cuda()
                output, outputs = model(inputs,return_feats=True)
                acc = (outputs.argmax(dim=1) == labels.cuda()).float().mean()
                acces_eval.update(acc.item(), inputs.size(0))
            eval_accs.append(acces_eval.avg)
            logger.info(f"Epoch [{epoch}/{cfg.TEST.EPOCH}] acc: {acces_eval.avg * 100:.2f}%")
            model.train()
            if acces_eval.avg >= max(eval_accs):
                best_epoch = epoch
    if type == "val":
        logger.info(f"Best epoch: {best_epoch}, acc: {max(eval_accs) * 100:.2f}%")
        return max(eval_accs)
    else:
        return acces_eval.avg
