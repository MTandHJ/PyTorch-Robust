

from typing import Optional, Tuple, List, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from .utils import getLogger



class IdentityTransform:

    def __call__(self, x: Any) -> Any:
        return x

class OrderTransform:

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, data: Tuple) -> List:
        return [transform(item) for item, transform in zip(data, self.transforms)]


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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        data = self.data[index]
        return self.transforms(data)


AUGMENTATIONS = {
    'none' : IdentityTransform(),
    'tensor': T.ToTensor(),
    'cifar': T.Compose((
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        )),
}

