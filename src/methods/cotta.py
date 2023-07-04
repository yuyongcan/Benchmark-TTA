from copy import deepcopy

import torch
import torch.jit
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm

from src.data.augmentations import get_tta_transforms
from src.utils.utils import  deepcopy_model


def update_ema_variables(ema_model, model, alpha_teacher):  # , iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, mt_alpha, rst_m, ap, dataset_name, steps=1, episodic=False, num_aug=32):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(dataset_name)
        self.softmax_entropy = softmax_entropy_cifar if "cifar" in dataset_name else softmax_entropy_imagenet
        self.num_aug = num_aug

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)
        # use this line if you want to reset the teacher model as well. Maybe you also 
        # want to del self.model_ema first to save gpu memory.
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            self.copy_model_and_optimizer(self.model, self.optimizer)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, optimizer):
        outputs = self.model(x)
        self.model_ema.train()
        # Teacher Prediction
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        outputs_emas = []
        to_aug = anchor_prob.mean(0) < self.ap
        if to_aug:
            for i in range(self.num_aug):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        loss = (self.softmax_entropy(outputs, outputs_ema.detach())).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        if self.rst > 0:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < self.rst).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        return outputs_ema

    @staticmethod
    def collect_params(model):
        """Collect all trainable parameters.

        Walk the model's modules and collect all parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if True:  # isinstance(m, nn.BatchNorm2d): collect all
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        # print(nm, np)
        return params, names

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())

        model_anchor = deepcopy_model(model)

        optimizer_state = deepcopy(optimizer.state_dict())

        ema_model = deepcopy_model(model)

        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor

    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what we update
        model.requires_grad_(False)
        # enable all trainable
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(True)
        return model

    @staticmethod
    def check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: " \
                               "check which require grad"
        assert not has_all_params, "tent should not update all params: " \
                                   "check which require grad"
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema):  # -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_cifar(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
