

from typing import Union, Iterable, Optional
import torch
import torch.nn.functional as F

import math

from models.base import AdversarialDefensiveModule
from .config import DEVICE


class PGD_TE(AdversarialDefensiveModule):
    def __init__(
        self, num_samples: int = 50000, num_classes: int = 10, momentum: float = 0.9, 
        reg_weight: float = 300., start_es: int = 90, end_es: int = 150, device =  DEVICE
    ):
        super().__init__()
        # initialize soft labels to ont-hot vectors
        self.register_buffer(
            "soft_labels", torch.zeros(num_samples, num_classes, dtype=torch.float).to(device)
        )
        self.momentum = momentum
        self.reg_weight = reg_weight
        self.start_es = start_es
        self.end_es = end_es

    def ema(self, index: torch.Tensor, prob: torch.Tensor) -> None:
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

    def get_soft_targets(self, logits: torch.Tensor, index: int, epoch: int) -> torch.Tensor:
        epoch += 1 # epoch starts from 0
        if epoch >= self.start_es:
            prob = F.softmax(logits.detach(), dim=1)
            self.ema(index, prob) # update
            soft_targets = self.soft_labels[index] / self.soft_labels[index].sum(1, keepdim=True)
        else:
            soft_targets = None
        return soft_targets

    def sigmoid_rampup(self, current: int) -> float:
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        current += 1 # epoch starts from 0
        if current < self.start_es:
            return 0.
        if current > self.end_es:
            return self.reg_weight
        else:
            phase = 1.0 - (current - self.start_es) / (self.end_es - self.start_es)
            return math.exp(-5.0 * phase * phase) * self.reg_weight
    
    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None, weight: float = 0.
    ) -> torch.Tensor:
        if soft_targets:
            return cross_entropy(logits, labels) + weight * ((F.softmax(logits, dim=-1) - soft_targets) ** 2).mean()
        else:
            return cross_entropy(logits, labels)


class TRADES_TE(PGD_TE):

    def forward(
        self, logits_adv: torch.Tensor, logits_nat: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None, 
        beta: float = 6., weight: float = 0.
    ) -> torch.Tensor:
        if soft_targets:
            return beta * kl_divergence(logits_adv, logits_nat) + weight * ((F.softmax(logits_nat, dim=-1) - soft_targets) ** 2).mean()
        else:
            return beta * kl_divergence(logits_adv, logits_nat)


def cross_entropy(
    outs: torch.Tensor, 
    labels: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    """
    cross entropy with logits
    """
    return F.cross_entropy(outs, labels, reduction=reduction)

def cross_entropy_softmax(
    probs: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    cross entropy with probs
        probs: the softmax of logits
    """
    return F.nll_loss(probs.log(), labels, reduction=reduction)

def kl_divergence(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "batchmean"
) -> torch.Tensor:
    # KL divergence
    assert logits.size() == targets.size()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)

def mse_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    return F.mse_loss(inputs, targets, reduction=reduction)

def lploss(
    x: torch.Tensor,
    p: Union[int, float, 'fro', 'nuc'] = 'fro',
    dim: Union[int, Iterable] = -1
):
    return torch.norm(x, p=p, dim=dim).mean()
