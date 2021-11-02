



from typing import Tuple, Optional, Callable
import torch
import torch.nn.functional as  F
import eagerpy as ep
from foolbox.attacks import LinfProjectedGradientDescentAttack

from models.base import ADArch
from .loss_zoo import cross_entropy, kl_divergence


def enter_attack_exit(func) -> Callable:
    def wrapper(attacker: "Adversary", *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class LinfPGDKLDiv(LinfProjectedGradientDescentAttack):

    # kl divergence as the loss function ...
    def get_loss_fn(self, model, logits_p):
        def loss_fn(inputs):
            logits_q = model(inputs)
            return ep.kl_div_with_logits(logits_p, logits_q).sum()
        return loss_fn

class LinfPGDSoftmax(LinfProjectedGradientDescentAttack):

    # the model returns the probs after softmax ...
    def get_loss_fn(self, model, labels):
        def loss_fn(inputs):
            probs = model(inputs)
            loss = F.nll_loss(probs.log().raw, labels.raw)
            return ep.astensor(loss)
        return loss_fn



class LinfFriendlyPGD:

    def __init__(
        self,
        model: ADArch,
        rel_stepsize: float = 0.25,
        abs_stepsize: Optional[float] = None,
        steps: int = 10, epsilon: float = 8 / 255,
        tau: int = 1, omega: float = 0.,
        random_start: bool = True,
        random_type: str = 'uniform',
        dynamictau: bool = True,
        bounds: Tuple[float] = (0, 1)
    ):
        if abs_stepsize is None:
            self.stepsize = rel_stepsize * epsilon
        else:
            self.stepsize = abs_stepsize
        self.model = model
        self.steps = steps
        self.epsilon = epsilon
        self.tau = tau
        self.omega = omega
        self.random_start = random_start
        self.random_type = random_type
        self.dynamictau = dynamictau
        self.bounds = bounds

    def adjust_tau(self, epoch: int):
        epoch = epoch + 2 # FAT starts with epoch == 1 and adjust it before each iteration
        if self.dynamictau:
            if epoch <= 50:
                self.tau = 0
            elif epoch <= 90:
                self.tau = 1
            else:
                self.tau = 2

    def startrandom(self, x: torch.Tensor):
        if self.random_type == 'uniform':
            eta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        elif self.random_type == 'normal':
            eta = torch.randn_like(x) * 0.001
        else:
            raise ValueError(f"No such random type {self.random_type}")
        return x + eta

    def loss_fn(
        self, cur_logits: torch.Tensor, 
        labels: torch.Tensor, logits_nat: Optional[torch.Tensor] = None
    ):
        return cross_entropy(cur_logits, labels, reduction='mean')

    def project(self, adv: torch.Tensor, source: torch.Tensor):
        return source + torch.clamp(adv - source, -self.epsilon, self.epsilon)

    @enter_attack_exit
    def attack(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        logits_nat: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:

        self.model.eval()
        x = inputs.clone()

        if self.random_start:
            x = self.startrandom(x)
            x = torch.clamp(x, *self.bounds)
        
        taus = torch.ones_like(labels) * self.tau
        indices = torch.ones_like(labels).astype(torch.bool)

        for _ in range(self.steps):
            cur_x = x[indices].clone().requires_grad_(True)
            cur_logits = self.model(cur_x)
            cur_pred = cur_logits.argmax(dim=1) != labels[indices]
            taus[indices][cur_pred] -= 1
            cur_taus = (taus > 0)[indices]
            indices[taus <= 0] = False
            loss = self.loss_fn(cur_logits[cur_taus], labels[indices], logits_nat)
            loss.backward()
            gradients = cur_x[cur_taus].grad.sign()
            x[indices] = x[indices] + self.stepsize * gradients + self.omega * torch.randn_like(x[indices])
            x[indices] = self.project(x[indices], inputs[indices])
            x[indices] = torch.clamp(x[indices], **self.bounds)

        return x

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)


class LinfFrendlyPGDKL(LinfFriendlyPGD):

    def loss_fn(
            self, cur_logits: torch.Tensor, 
            labels: torch.Tensor, logits_nat: Optional[torch.Tensor] = None
        ):
            return kl_divergence(cur_logits, logits_nat, reduction='sum')

    def adjust_tau(self, epoch: int):
        epoch = epoch + 2 # FAT starts with epoch == 1 and adjust it before each iteration
        if self.dynamictau:
            if epoch <= 30:
                self.tau = 0
            elif epoch <= 50:
                self.tau = 1
            elif epoch <= 70:
                self.tau = 2
            else:
                self.tau = 3









