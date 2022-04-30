

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

def cross_entropy_with_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    cross entropy with probs
        probs: the softmax of logits
    """
    inputs = F.log_softmax(logits, dim=-1)
    loss = inputs * targets
    if reduction == "sum":
        loss = loss.sum(-1).sum().neg()
    elif reduction == "mean":
        loss = loss.sum(-1).mean().neg()
    elif reduction == "none":
        loss.neg()
    else:
        raise ValueError(f"reduction should be in ['sum', 'mean', 'none'] but {reduction} received ...")
    return loss


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
