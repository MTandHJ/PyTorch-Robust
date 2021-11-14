


## Cutout


> [DeVries T. and Taylor G. W. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.](https://arxiv.org/abs/1708.04552)

[official-code](https://github.com/uoguelph-mlrg/Cutout)


## Usage

### CIFAR-10

	python STD.py resnet18 cifar10 --n-holes=1 --length=16 -lp=Cutout
	python STD.py wrn_28_10 cifar10 --n-holes=1 --length=16 -lp=Cutout


### CIFAR-100

	python STD.py resnet18 cifar100 --n-holes=1 --length=8 -lp=Cutout
	python STD.py wrn_28_10 cifar100 --n-holes=1 --length=8 -lp=Cutout


### SVHN

**Note:** The official code employs WideResNet-16-8 to train on SVHN including the extra data.

	python STD.py wrn_28_10 svhn --n-holes=1 --length=20 --epochs=160 -lr=0.01 -lp=Cutout-svhn



## Evaluation



### CIFAR-10



|      |   Net    |      | Clean | CIFAR-10-C |
| :--: | :------: | :--: | :---: | :--------: |
|  AT  | ResNet18 | last | 96.09 |            |



