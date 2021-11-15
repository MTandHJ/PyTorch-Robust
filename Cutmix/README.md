
## Cutmix


> [Yun S., Han D., Oh S., Chu S., Choe J. and Yoo Y. CutMix: Regularization strategy to train strong classifiers with localizable features. In International Conference on Computer Vision (ICCV), 2019.](https://arxiv.org/abs/1905.04899)

[official-code](https://github.com/clovaai/CutMix-PyTorch)


## Usage

**Note:** The follow config is for PyramidNet-200 but we employs ResNet18 here.

### CIFAR-10

	python STD.py resnet18 cifar10 --alpha=1 --cutmix-prob=0.5 -wd=1e-4 -lr=0.25 -lp=Cutmix --epochs=300 -b=64

### CIFAR-100

	python STD.py resnet18 cifar100 --alpha=1 --cutmix-prob=0.5 -wd=1e-4 -lr=0.25 -lp=Cutmix --epochs=300 -b=64

