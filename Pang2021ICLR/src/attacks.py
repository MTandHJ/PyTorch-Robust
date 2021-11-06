

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .loss_zoo import cross_entropy, kl_divergence




class BasePGD:

    def __init__(
        self, epsilon: float, steps: int, stepsize: float,
        random_start: bool = True, bounds: Tuple[float] = (0., 1.)
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.stepsize = stepsize
        self.random_start = random_start
        self.bounds = bounds

    def atleast_kd(self, x: torch.Tensor, k: int) -> torch.Tensor:
        size = x.size() + (1,) * (k - x.ndim)
        return x.view(size)

    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def project(self, adv: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def calc_grad(self, model: nn.Module, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        logits = model(x)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        return x.grad

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        x0 = inputs.clone()

        if self.random_start:
            x = x0 + self.get_random_start(x0)
            x = torch.clamp(x, *self.bounds)
        else:
            x = x0

        for _ in range(self.steps):
            grad = self.calc_grad(model, x, targets)
            x = x + self.stepsize * self.normalize(grad)
            x = self.project(x, x0)
            x = torch.clamp(x, *self.bounds)

        return x

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)


class LinfPGD(BasePGD):

    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x0).uniform_(-self.epsilon, self.epsilon)

    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.sign()

    def project(self, adv: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return source + torch.clamp(adv - source, -self.epsilon, self.epsilon)

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, targets, 'sum')


class L2PGD(BasePGD):
    EPS = 1e-12
    def get_random_start(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.randn_like(x.flatten(1))
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x.ndim)
        r = torch.rand_like(n) # r = 1 for some implementations
        delta *= r / n * self.epsilon
        return delta

    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        norms = grad.flatten(1).norm(p=2, dim=1)
        norms = self.atleast_kd(norms, grad.ndim)
        grad /= (norms + self.EPS)
        return grad

    def project(self, adv: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return source + torch.renorm(adv - source, p=2, dim=0, maxnorm=self.epsilon)


class LinfPGDKLdiv(LinfPGD):

    def get_random_start(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.randn_like(x) * 0.001 # TRADES adopts normal distribution
        return delta

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return kl_divergence(logits, targets, reduction='sum')

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        x0 = inputs.clone()

        if self.random_start:
            x = x0 + self.get_random_start(x0)
            # x = torch.clamp(x, *self.bounds)
        else:
            x = x0

        for _ in range(self.steps):
            grad = self.calc_grad(model, x, targets)
            x = x + self.stepsize * self.normalize(grad)
            x = self.project(x, x0)
            x = torch.clamp(x, *self.bounds)

        return x

class L2PGDKLdiv(L2PGD):

    def get_random_start(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.randn_like(x.flatten(1)) * 0.001
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x.ndim)
        r = 1. # r = 1 for some implementations
        delta *= r / n * self.epsilon
        return delta

    def normalize(self, adv: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        norms = grad.flatten(1).norm(p=2, dim=1)
        norms = self.atleast_kd(norms, grad.ndim)
        grad /= norms
        if (norms == 0).any():
            grad[norms == 0].normal_()
        return grad

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return kl_divergence(logits, targets, reduction='sum')

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        x0 = inputs.clone()

        if self.random_start:
            x = x0 + self.get_random_start(x0)
            # x = torch.clamp(x, *self.bounds)
        else:
            x = x0

        for _ in range(self.steps):
            grad = self.calc_grad(model, x, targets)
            x = x + self.stepsize * self.normalize(grad)
            x = self.project(x, x0)
            x = torch.clamp(x, *self.bounds)

        return x














