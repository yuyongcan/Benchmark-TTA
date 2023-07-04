"""
Builds upon: https://github.com/zhangmarvin/memo
Corresponding paper: https://arxiv.org/abs/2110.09506
"""
from copy import deepcopy

import numpy as np
import torch
import torch.jit
from PIL import Image
from torch import nn

from ..data.augmentations import aug_cifar, aug_imagenet


def tta(image, n_augmentations, aug):
    image = np.clip(image[0].cpu().numpy() * 255., 0, 255).astype(np.uint8).transpose(1, 2, 0)
    inputs = [aug(Image.fromarray(image)) for _ in range(n_augmentations)]
    inputs = torch.stack(inputs).cuda()
    return inputs


class MEMO(nn.Module):
    """MEMO
    """

    def __init__(self, model, optimizer, steps, episodic, n_augmentations, dataset_name):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.n_augmentations = n_augmentations
        self.augmentations = aug_cifar if "cifar" in dataset_name else aug_imagenet

        self.models = [self.model]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def forward(self, x):
        origin_x = x[0]
        if self.episodic:
            self.reset()
        self.batch_size = x[0].shape[0]
        for _ in range(self.steps):
            x_aug = torch.concat([input for input in x[1:]], dim=0)

            _ = self.forward_and_adapt(x_aug)

        return self.model(origin_x)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        self.optimizer.zero_grad()
        for i in range(self.batch_size):
            x_aug_t = x[i::self.batch_size, :, :, :]
            outputs = self.model(x_aug_t)
            loss, _ = marginal_entropy(outputs)
            loss /= self.batch_size
            loss.backward()
        self.optimizer.step()
        return outputs

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
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
