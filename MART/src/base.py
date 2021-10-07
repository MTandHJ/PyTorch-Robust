


from typing import Callable, TypeVar, Any, Union, Optional, List, Tuple, Dict, Iterable, cast
import torch
import torch.nn as nn
import foolbox as fb
import eagerpy as ep
import os

from models.base import AdversarialDefensiveModule
from .criteria import LogitsAllFalse
from .utils import AverageMeter, ProgressMeter, timemeter
from .loss_zoo import mart_loss
from .config import SAVED_FILENAME, BOUNDS, PREPROCESSING


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
        device: torch.device,
        loss_func: Callable, 
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy"
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc", fmt=".3%")
        self.progress = ProgressMeter(self.loss, self.acc)
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    @timemeter("MART/Epoch")
    def adv_train(
        self, 
        trainloader: Iterable[Tuple[torch.Tensor, torch.Tensor]], 
        attacker: "Adversary", 
        *, leverage: float = 6., epoch: int = 8888
    ) -> float:
    
        assert isinstance(attacker, Adversary)
        self.progress.step() # reset the meter
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            _, clipped, _ = attacker(inputs, labels)
            
            self.model.train()
            logits_nat = self.model(inputs)
            logits_adv = self.model(inputs)
            loss = mart_loss(logits_nat, logits_adv, labels, leverage=leverage)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            accuracy_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(accuracy_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step() # update the learning rate
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



