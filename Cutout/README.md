


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



|      |   Net    |      | Clean |
| :--: | :------: | :--: | :---: |
|  AT  | ResNet18 | last | 96.09 |



|brightness | defocus_blur | fog | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate | snow | speckle_noise | contrast | elastic_transform | frost | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur|
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|94.698| 81.546| 89.816| 66.970| 62.618| 79.138| 78.608| 93.010| 85.144| 56.670| 80.052| 86.650| 79.334| 40.754| 50.974| 74.496| 52.340| 87.704| 75.572|

