


## AugMix

> [Hendrycks D., Norman M., Ekin D. C., Barret Z., Justin G. and Balaji L. AugMix: A simple data processing method to improve robustness and uncertainty. In International Conference on Learning Representations (ICLR), 2020.](https://arxiv.org/pdf/1912.02781.pdf)

[official-code](https://github.com/google-research/augmix)

## Usage

### CIFAR-10

	python STD.py resnet18 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 -epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 -epochs=100 -lp=AugMix

### CIFAR-100

	python STD.py resnet18 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 -epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 -epochs=100 -lp=AugMix

