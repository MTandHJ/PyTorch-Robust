

from typing import Any, Callable
import torch
import torch.nn as nn
import abc


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

class ADArch(AdversarialDefensiveModule):

    def set_normalizer(self, normalizer: Callable) -> None:
        self.normalizer = normalizer

    def __call__(self, inputs: torch.Tensor, **kwargs: Any) -> Any:
        try:
            inputs = self.normalizer(inputs)
        except AttributeError as e:
            errors = str(e) + "\n >>> You shall set normalizer manually by calling 'set_normalizer' ..."
            raise AttributeError(errors)
        return super().__call__(inputs, **kwargs)

class DataParallel(nn.DataParallel, AdversarialDefensiveModule): ...


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

