


from typing import Callable, TypeVar, Any, Union, Optional, List, Tuple, Dict, Iterable, cast
import torch
import torch.nn as nn
import foolbox as fb
import eagerpy as ep
import copy
import os

from models.base import AdversarialDefensiveModule
from .criteria import LogitsAllFalse
from .utils import AverageMeter, ProgressMeter, timemeter, getLogger
from .loss_zoo import cross_entropy, kl_divergence, lploss
from .config import SAVED_FILENAME, PRE_BESTNAT, PRE_BESTROB, BOUNDS, PREPROCESSING


def enter_attack_exit(func) -> Callable:
    def wrapper(attacker: "Adversary", *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Coach:
    
    def __init__(
        self, model: AdversarialDefensiveModule,
        proxy: AdversarialDefensiveModule,
        device: torch.device,
        awp_gamma: float,
        awp_warmup: int,
        loss_func: Callable, 
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy"
    ):
        self.model = model
        self.proxy = proxy
        self.device = device
        self.awp_gamma = awp_gamma
        self.awp_warmup = awp_warmup
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.proxy_optimizer = torch.optim.SGD(proxy.parameters(), lr=0.01)
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc", fmt=".3%")
        self.progress = ProgressMeter(self.loss, self.acc)

        self._best_nat = 0.
        self._best_rob = 0.

    def save_best_nat(self, acc_nat: float, path: str, prefix: str = PRE_BESTNAT):
        if acc_nat > self._best_nat:
            self._best_nat = acc_nat
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0
    
    def save_best_rob(self, acc_rob: float, path: str, prefix: str = PRE_BESTROB):
        if acc_rob > self._best_rob:
            self._best_rob = acc_rob
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0

    def check_best(
        self, acc_nat: float, acc_rob: float,
        path: str, epoch: int = 8888
    ):
        logger = getLogger()
        if self.save_best_nat(acc_nat, path):
            logger.debug(f"[Coach] Saving the best nat ({acc_nat:.3%}) model at epoch [{epoch}]")
        if self.save_best_rob(acc_rob, path):
            logger.debug(f"[Coach] Saving the best rob ({acc_rob:.3%}) model at epoch [{epoch}]")
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    @torch.no_grad()
    def diff_in_weights(self):
        diff_dict = dict()
        model_state_dict = self.model.state_dict()
        proxy_state_dict = self.proxy.state_dict()
        for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
            if len(old_w.size()) < 1:
                continue
            if 'weight' in old_k:
                diff_w = new_w.data - old_w.data # v
                diff_dict[old_k] = old_w.data.norm() / (diff_w.norm() + 1e-20) * diff_w # v / \|v\| * \|w\|
        return diff_dict
    
    @torch.no_grad()
    def add_into_weights(self, diff: Dict, coeff: float):
        _keys = diff.keys()
        for name, params in self.model.named_parameters():
            if name in _keys:
                params.data.add_(diff[name] * coeff)

    def perturb(self, diff: Dict):
        self.add_into_weights(diff, self.awp_gamma)
    
    def restore(self, diff: Dict):
        self.add_into_weights(diff, -self.awp_gamma)

    def calc_awp_adv(self, clipped: torch.Tensor, labels: torch.Tensor) -> Dict:
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        outs = self.proxy(clipped)
        loss = self.loss_func(outs, labels).neg() # neg !

        self.proxy_optimizer.zero_grad()
        loss.backward()
        self.proxy_optimizer.step()
        
        return self.diff_in_weights()

    @timemeter("AT-AWP/Epoch")
    def adv_train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        attacker: "Adversary", 
        *, epoch: int = 8888
    ) -> float:
    
        assert isinstance(attacker, Adversary)
        self.progress.step() # reset the meter
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            _, clipped, _ = attacker(inputs, labels)

            # perturb the model
            if epoch >= self.awp_warmup:
                diff = self.calc_awp_adv(clipped, labels)
                self.perturb(diff)
            
            self.model.train()
            outs = self.model(clipped)
            loss = self.loss_func(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # restore the model
            if epoch >= self.awp_warmup:
                self.restore(diff)

            accuracy_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(accuracy_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step() # update the learning rate
        return self.loss.avg

    def calc_awp_trades(
        self, inputs_nat: torch.Tensor, 
        inputs_adv: torch.Tensor, 
        labels: torch.Tensor, leverage: float
    ) -> Dict:
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        logits_nat = self.proxy(inputs_nat)
        logits_adv = self.proxy(inputs_adv)
        loss_nat = cross_entropy(logits_nat, labels)
        loss_adv = kl_divergence(logits_adv, logits_nat)
        loss = (loss_nat + leverage * loss_adv).neg() # neg !
        
        self.proxy_optimizer.zero_grad()
        loss.backward()
        self.proxy_optimizer.step()
        
        return self.diff_in_weights()


    @timemeter("TRADES-AWP/Epoch")
    def trades(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        attacker: "Adversary", 
        *, leverage: float = 6., epoch: int = 8888
    ) -> float:

        self.progress.step()
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                self.model.eval()
                logits = self.model(inputs).detach()
            criterion = LogitsAllFalse(logits) # perturbed by kl loss
            _, inputs_adv, _ = attacker(inputs, criterion)

            # perturb the model
            if epoch >= self.awp_warmup:
                diff = self.calc_awp_trades(
                    inputs, inputs_adv, labels, leverage
                )
                self.perturb(diff)
            
            self.model.train()
            logits_nat = self.model(inputs)
            logits_adv = self.model(inputs_adv)
            loss_nat = cross_entropy(logits_nat, labels)
            loss_adv = kl_divergence(logits_adv, logits_nat)
            loss = loss_nat + leverage * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # restore the model
            if epoch >= self.awp_warmup:
                self.restore(diff)

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg


class FBDefense:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        bounds: Tuple[float, float] = BOUNDS, 
        preprocessing: Optional[Dict] = PREPROCESSING
    ) -> None:
        self.rmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device            
        )

        self.model = model
    
    def train(self, mode: bool = True) -> None:
        self.model.train(mode=mode)

    def eval(self) -> None:
        self.train(mode=False)

    def query(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rmodel(inputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.query(inputs)


class Adversary:
    """
    Adversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model: AdversarialDefensiveModule, 
        attacker: Callable, device: torch.device,
        epsilon: Union[None, float, List[float]],
        bounds: Tuple[float, float] = BOUNDS, 
        preprocessing: Optional[Dict] = PREPROCESSING
    ) -> None:

        model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    def attack(
        self, 
        inputs: torch.Tensor, 
        criterion: Any, 
        epsilon: Union[None, float, List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluation mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    def __call__(
        self, 
        inputs: torch.Tensor, criterion: Any,
        epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.attack(inputs, criterion, epsilon)


class AdversaryForTrain(Adversary):

    @enter_attack_exit
    def attack(
        self, inputs: torch.Tensor, 
        criterion: Any, 
        epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return super(AdversaryForTrain, self).attack(inputs, criterion, epsilon)


class AdversaryForValid(Adversary): 

    @torch.no_grad()
    def accuracy(self, inputs: torch.Tensor, labels: torch.Tensor) -> int:
        inputs_, labels_ = ep.astensors(inputs, labels)
        del inputs, labels

        self.model.eval() # make sure in evaluation mode ...
        predictions = self.fmodel(inputs_).argmax(axis=-1)
        accuracy = (predictions == labels_)
        return cast(int, accuracy.sum().item())

    def evaluate(
        self, 
        dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        epsilon: Union[None, float, List[float]] = None,
        *, defending: bool = True
    ) -> Tuple[float, float]:

        datasize = len(dataloader.dataset) # type: ignore
        acc_nat = 0
        acc_adv = 0
        self.model.defend(defending) # enter 'defending' mode
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            _, _, is_adv = self.attack(inputs, labels, epsilon)
            acc_nat += self.accuracy(inputs, labels)
            acc_adv += (~is_adv).sum().item()
        return acc_nat / datasize, acc_adv / datasize



