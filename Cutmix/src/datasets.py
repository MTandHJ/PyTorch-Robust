

from typing import Callable, Optional, Tuple, List, Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import os
from PIL import Image

from .utils import getLogger
from .config import ROOT


class Compose(T.Compose):

    def __init__(self, transforms: List):
        super().__init__(transforms)
        assert isinstance(transforms, list), f"List of transforms required, but {type(transforms)} received ..."

    def append(self, transform: Callable):
        self.transforms.append(transform)

class IdentityTransform:

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, x: Any) -> Any:
        return x

# Reference: https://github.com/clovaai/CutMix-PyTorch
class CutmixTransform:

    def __init__(self, alpha: float=1., cutmix_prob: float = 1.) -> None:
        self.alpha = alpha
        self.cutmix_prob = cutmix_prob

    def _rand_bbox(self, H: int, W: int, lam: float):
        cut_rate = np.sqrt(1 - lam)
        cut_h = np.int(cut_rate * H)
        cut_w = np.int(cut_rate * W)

        center_h = np.random.randint(0, H)
        center_w = np.random.randint(0, W)

        lu_h = np.clip(center_h - cut_h // 2, 0, H)
        lu_w = np.clip(center_w - cut_w // 2, 0, W)
        rb_h = np.clip(center_h + cut_h // 2, 0, H)
        rb_w = np.clip(center_w + cut_w // 2, 0, W)

        return lu_h, lu_w, rb_h, rb_w

    def cutmix(self, inputs: torch.Tensor, labels: torch.Tensor):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        if lam >= self.cutmix_prob:
            return inputs, labels, labels, lam

        inputs = inputs.clone()
        batch_size, _, H, W = inputs.size()
        index = torch.randperm(batch_size)

        lu_h, lu_w, rb_h, rb_w = self._rand_bbox(H, W, lam)
        inputs[:, :, lu_h:rb_h, lu_w:rb_w] = inputs[index, :, lu_h:rb_h, lu_w:rb_w]
        y_a, y_b = labels, labels[index]
        lam = 1 - (rb_h - lu_h) * (rb_w - lu_w) / (H * W)
        return inputs, y_a, y_b, lam

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, cutmix_prob={self.cutmix_prob})"

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor) -> Any:
        return self.cutmix(inputs, labels)

class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def append(self, transform: Callable, index: int = 0):
        self.transforms[index].append(transform)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '['
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n]'
        return format_string

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]


# https://github.com/VITA-Group/Adversarial-Contrastive-Learning/blob/937019219497b449f4cb61cc6118fdf32cc3de12/data/cifar10_c.py#L9
class CIFAR10C(Dataset):
    filename = "CIFAR-10-C"
    def __init__(
        self, root: str = ROOT, 
        transform: Optional[Callable] = None, 
        corruption_type: str = 'snow'
    ):
        root = os.path.join(root, self.filename)
        dataPath = os.path.join(root, '{}.npy'.format(corruption_type))
        labelPath = os.path.join(root, 'labels.npy')

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]
        

class CIFAR100C(CIFAR10C):
    filename = "CIFAR-100-C"



class WrapperSet(Dataset):

    def __init__(
        self, dataset: Dataset,
        transforms: Optional[str] = None
    ) -> None:
        """
        Args:
            dataset: dataset;
            transforms: string spilt by ',', such as "tensor,none'
        """
        super().__init__()

        self.data = dataset

        try:
            counts = len(self.data[0])
        except IndexError:
            getLogger().info("[Dataset] zero-size dataset, skip ...")
            return

        if transforms is None:
            transforms = ['none'] * counts
        else:
            transforms = transforms.split(',')
        self.transforms = [AUGMENTATIONS[transform] for transform in transforms]
        if counts == 1:
            self.transforms = self.transforms[0]
        else:
            self.transforms = OrderTransform(self.transforms)
        getLogger().info(self.transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)


AUGMENTATIONS = {
    'none' : Compose([IdentityTransform()]),
    'tensor': Compose([T.ToTensor()]),
    'cifar': Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
    ]),
}

