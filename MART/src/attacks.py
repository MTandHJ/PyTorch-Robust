

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .loss_zoo import cross_entropy, kl_divergence




class BasePGD:

    def __init__(
        self,
        epsilon: float, steps: int, stepsize: float,
        random_start: bool = True,
        bounds: Tuple[float] = (0, 1)
    ):
        self.epsilon = epsilon
        self.steps = steps
        self.stepsize = stepsize
        self.random_start = random_start
        self.bounds = bounds

    def atleast_kd(self, x: torch.Tensor, k: int) -> torch.Tensor:
        size = x.size() + (1,) * (k - x.ndim)
        return x.view(size)

    def get_random_start(self, x: torch.Tensor):
        raise NotImplementedError
    
    def normalize(self, adv: torch.Tensor, grad: torch.Tensor):
        raise NotImplementedError

    def project(self, adv: torch.Tensor, source: torch.Tensor):
        raise NotImplementedError

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError

    def attack(
        self, model: nn.Module, 
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        x = inputs.clone()
        if self.random_start:
            x = self.get_random_start(x)
            x = torch.clamp(x, *self.bounds)

        for _ in range(self.steps):
            x.requires_grad_(True)
            logits = model(x)
            loss = self.loss_fn(logits, targets)
            grad = torch.autograd.grad(loss, x)[0].detach()
            x = self.normalize(x.detach(), grad)
            x = self.project(x, inputs)
            x = torch.clamp(x, *self.bounds)
        return x.detach()

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)



class LinfPGD(BasePGD):

    def get_random_start(self, x: torch.Tensor):
        delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        return x + delta

    def normalize(self, adv: torch.Tensor, grad: torch.Tensor):
        return adv + self.stepsize * grad.sign()

    def project(self, adv: torch.Tensor, source: torch.Tensor):
        return source + torch.clamp(adv - source, -self.epsilon, self.epsilon)

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return cross_entropy(logits, targets, reduction='mean')

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)


class L2PGD(BasePGD):
    EPS = 1e-12
    def get_random_start(self, x: torch.Tensor):
        delta = torch.randn_like(x.flatten(1))
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x.ndim)
        r = torch.rand_like(n) # r = 1 for some implementations
        delta *= r / n * self.epsilon
        return x + delta

    def normalize(self, adv: torch.Tensor, grad: torch.Tensor):
        norms = grad.flatten(1).norm(p=2, dim=1)
        norms = self.atleast_kd(norms, grad.ndim)
        grad /= (norms + self.EPS)
        return adv + self.stepsize * grad

    def project(self, adv: torch.Tensor, source: torch.Tensor):
        return source + torch.renorm(adv - source, p=2, dim=0, maxnorm=self.epsilon)


class LinfPGDKLdiv(LinfPGD):

    def get_random_start(self, x: torch.Tensor):
        delta = torch.randn_like(x) * 0.001 # TRADES adopts normal distribution
        return x + delta

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return kl_divergence(logits, targets, reduction='sum')

    def attack(
        self, model: nn.Module, 
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        x = inputs.clone()
        if self.random_start:
            x = self.get_random_start(x)
            # TRADES excludes the clip operation at first
            # x = torch.clamp(x, *self.bounds) 

        for _ in range(self.steps):
            x.requires_grad_(True)
            logits = model(x)
            loss = self.loss_fn(logits, targets)
            grad = torch.autograd.grad(loss, x)[0].detach()
            x = self.normalize(x.detach(), grad)
            x = self.project(x, inputs)
            x = torch.clamp(x, *self.bounds)
        return x.detach()


class L2PGDKLdiv(L2PGD):

    def get_random_start(self, x: torch.Tensor):
        delta = torch.randn_like(x.flatten(1)) * 0.001
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x.ndim)
        r = 1. # r = 1 for some implementations
        delta *= r / n * self.epsilon
        return x + delta

    def normalize(self, adv: torch.Tensor, grad: torch.Tensor):
        norms = grad.flatten(1).norm(p=2, dim=1)
        norms = self.atleast_kd(norms, grad.ndim)
        grad /= norms
        if (norms == 0).any():
            grad[norms == 0].normal_()
        return adv + self.stepsize * grad

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        return kl_divergence(logits, targets, reduction='sum')

    def attack(
        self, model: nn.Module, 
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        x = inputs.clone()
        if self.random_start:
            x = self.get_random_start(x)
            # TRADES excludes the clip operation at first
            # x = torch.clamp(x, *self.bounds) 

        for _ in range(self.steps):
            x.requires_grad_(True)
            logits = model(x)
            loss = self.loss_fn(logits, targets)
            grad = torch.autograd.grad(loss, x)[0].detach()
            x = self.normalize(x.detach(), grad)
            x = self.project(x, inputs)
            x = torch.clamp(x, *self.bounds)
        return x.detach()
















