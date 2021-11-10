

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
    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        delta = torch.randn_like(x0.flatten(1))
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x0.ndim)
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

    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x0) * 0.001 # TRADES adopts normal distribution

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

    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        delta = torch.randn_like(x0.flatten(1)) * 0.001
        n = delta.norm(p=2, dim=1)
        n = self.atleast_kd(n, x0.ndim)
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


class FriendlyPGD(LinfPGD):

    def __init__(
        self, epsilon: float, steps: int, stepsize: float, tau: int = 0,
        random_start: bool = True, bounds: Tuple[float] = (0, 1)
    ) -> None:
        super().__init__(epsilon, steps, stepsize, random_start=random_start, bounds=bounds)

        self.tau = tau

    def adjust_tau(self, epoch: int):
        # The epoch of FAT starts with 1, but 0 of ours.
        # Therefore, use < instead of <= .
        if epoch < 50:
            self.tau = 0
        elif epoch < 90:
            self.tau = 1
        else:
            self.tau = 2

    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        # i.e. omega * torch.randn(iter_adv.shape).detach().cuda()
        # where omega is 0.001 for FAT.
        # This is for escaping the local minimum as the author suggests.
        return super().normalize(grad) + 0.001 * torch.rand_like(grad)

    @torch.no_grad()
    def which_to_attack(self, model: nn.Module, x: torch.Tensor, labels: torch.Tensor):
        logits = model(x)
        pred = logits.argmax(-1) != labels
        return pred.float().neg()

    def attack(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        x0 = inputs.clone()

        if self.random_start:
            x = x0 + self.get_random_start(x0)
            x = torch.clamp(x, *self.bounds)
        else:
            x = x0

        tau_counts = torch.ones(x0.size(0), device=x0.device) * self.tau

        for _ in range(self.steps):
            tau_counts += self.which_to_attack(model, x, targets)
            wta = tau_counts >= 0
            if wta.sum() == 0:
                break
            grad = self.calc_grad(model, x[wta], targets[wta])
            x[wta] = x[wta] + self.stepsize * self.normalize(grad)
            x = self.project(x, x0)
            x = torch.clamp(x, *self.bounds)

        return x


class FriendlyPGDKL(FriendlyPGD):

    def __init__(
        self, epsilon: float, steps: int, stepsize: float, tau: int = 0, 
        random_start: bool = True, bounds: Tuple[float] = (0, 1)
    ) -> None:
        super().__init__(epsilon, steps, stepsize, tau, random_start=random_start, bounds=bounds)

    def adjust_tau(self, epoch: int):
        # The epoch of FAT starts with 1, but 0 of ours.
        # Therefore, use < instead of <= .
        if epoch < 30:
            self.tau = 0
        elif epoch < 50:
            self.tau = 1
        elif epoch < 70:
            self.tau = 2
        else:
            self.tau = 3

    def get_random_start(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x0) * 0.001

    def normalize(self, grad: torch.Tensor) -> torch.Tensor:
        return grad.sign() # FAT-TRADES with omega == 0.0

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return kl_divergence(logits, targets, reduction='sum')

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        x0 = inputs.clone()

        with torch.no_grad():
            targets = model(x0)

        if self.random_start:
            x = x0 + self.get_random_start(x0)
            x = torch.clamp(x, *self.bounds)
        else:
            x = x0

        tau_counts = torch.ones(x0.size(0), device=x0.device) * self.tau

        for _ in range(self.steps):
            tau_counts += self.which_to_attack(model, x, labels)
            wta = tau_counts >= 0
            if wta.sum() == 0:
                break
            grad = self.calc_grad(model, x[wta], targets[wta])
            x[wta] = x[wta] + self.stepsize * self.normalize(grad)
            x = self.project(x, x0)
            x = torch.clamp(x, *self.bounds)

        return x









