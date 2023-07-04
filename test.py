from copy import deepcopy

import timm
from torch.nn.utils.weight_norm import WeightNorm

from src.data.data import *
from src.models.load_model import load_model
from src.utils import get_accuracy, AverageMeter
from train_source import cal_acc

# dataset, dataloader = loads
#
# # dataset, dataloader = load_dataset(dataset='officehome',root='/data2/yongcan.yu/datasets', adaptation='source', batch_size=4, workers=4,
# #                                    split='train',domain='art')
#
# # offcehome_domains = ['art', 'clipart', 'product', 'real']
# # model=load_model('officehome',domain='real')
# print()


# class Object:
#     pass
#
#
# x=Object()
# x.img=Object()
# x.img.size=1
# print(x)

# for i in range(10):
#     import datetime
#     print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# model_1=load_model('officehome_shot').cuda()
# model_1.load_state_dict(torch.load('./ckpt/models/officehome/RealWorld/model.pt'))
# model_2 = load_model('officehome_shot', checkpoint_dir='./ckpt/models', domain='RealWorld').cuda()
# model_1.eval()
# model_2.eval()
dataset, dataloader = load_dataset(dataset='imagenet_c', root='/data2/yongcan.yu/datasets', adaptation='source',
                                   batch_size=64, workers=4, domain='brightness', level=5, split='all',
                                   ckpt='./ckpt/Datasets')
model=timm.create_model('vit_base_patch16_224', pretrained=True).cuda()
# acc_1 = get_accuracy(model_1, data_loader=dataloader)
# acces = AverageMeter('acc')
# with torch.no_grad():
#     for i, data in enumerate(dataloader):
#         imgs, labels = data[0], data[1]
#         output = model_2(imgs.cuda())
#         predictions = output.argmax(1)
#         acc=(predictions == labels.cuda()).float().mean()
#         acces.update(acc.item(), imgs.size(0))
# print(acces.avg)
# for module in model_1.modules():
#     for _, hook in module._forward_pre_hooks.items():
#         if isinstance(hook, WeightNorm):
#             delattr(module, hook.name)
#
#
# model=deepcopy(model_1)
# for module in model.modules():
#     for _, hook in module._forward_pre_hooks.items():
#         if isinstance(hook, WeightNorm):
#             hook(module, 'weight')
#

for imgs, labels in dataloader:
    imgs=imgs.cuda()
    labels=labels.cuda()
    output=model(imgs)
    predictions=output.argmax(1)
    acc=(predictions==labels).float().mean()
    print(acc.item())
pass
