

from typing import Union, Iterable
import torch
import torch.nn.functional as F


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
    # targets = targets.clone().detach()
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

def mart_loss(
    logits_nat: torch.Tensor, logits_adv: torch.Tensor,
    labels: torch.Tensor, leverage: float = 6.
):
    order = torch.argsort(logits_adv, dim=1)[:, -2:]
    second = torch.where(order[:, -1] == labels, order[:, -2], order[:, -1])
    loss_pos = cross_entropy(logits_adv, labels) \
            + F.nll_loss((1.0001 - F.softmax(logits_adv, dim=1) + 1e-12), second)
    true_probs = torch.gather(
        F.softmax(logits_nat, dim=1), 1, labels.unsqueeze(1)
    ).squeeze()
    loss_neg = kl_divergence(logits_nat, logits_adv, reduction='none').sum(dim=1)
    loss_neg = (loss_neg * (1.0000001 - true_probs)).mean()
    return loss_pos + leverage * loss_neg
