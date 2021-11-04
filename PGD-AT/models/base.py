

from typing import Any, Callable
import torch
import torch.nn as nn
import abc

from src.config import DEVICE


class ADType(abc.ABC): ...

class AdversarialDefensiveModule(ADType, nn.Module):
    """
    Define some basic properties.
    """
    def __init__(self) -> None:
        super(AdversarialDefensiveModule, self).__init__()
        # Some model's outputs for training(evaluating) 
        # and attacking are different.
        self.attacking: bool = False
        self.defending: bool = True
        
    def attack(self, mode: bool = True) -> None:
        # enter attacking mode
        # for adversary only
        self.attacking = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.attack(mode)

    def defend(self, mode: bool = True) -> None:
        # enter defense mode
        # for some special techniques
        self.defending = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.defend(mode)


class DataParallel(nn.DataParallel, AdversarialDefensiveModule): ...

class ADArch(AdversarialDefensiveModule):

    def __init__(
        self, model: AdversarialDefensiveModule,
        mean: torch.Tensor, std: torch.Tensor, 
        device: torch.device = DEVICE
    ) -> None:
        super().__init__()
        if torch.cuda.device_count() > 1:
            self.model = DataParallel(model)
        else:
            self.model = model.to(device)

        self.mean, self.std = mean.to(device), std.to(device)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> Any:
        inputs = self._normalize(inputs)
        return  self.model(inputs, **kwargs)



if __name__ == "__main__":
    
    model = ADArch()
    model.child1 = AdversarialDefensiveModule()
    model.child2 = AdversarialDefensiveModule()
    model = DataParallel(model)

    print(model.attacking)
    model.attack(True)
    for m in model.children():
        print(m.attacking)

    model.defend(False)
    for m in model.children():
        print(m.defending)

