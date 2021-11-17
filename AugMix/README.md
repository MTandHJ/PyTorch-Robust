


## AugMix

> [Hendrycks D., Norman M., Ekin D. C., Barret Z., Justin G. and Balaji L. AugMix: A simple data processing method to improve robustness and uncertainty. In International Conference on Learning Representations (ICLR), 2020.](https://arxiv.org/pdf/1912.02781.pdf)

[official-code](https://github.com/google-research/augmix)

## Usage

### CIFAR-10

	python STD.py resnet18 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --epochs=100 -lp=AugMix

### CIFAR-100

	python STD.py resnet18 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --epochs=100 -lp=AugMix



## Evaluation



|      |   Net    |      | Clean  |
| :--: | :------: | :--: | :----: |
|  AT  | ResNet18 | last | 95.690 |



| brightness | defocus_blur | fog  | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate | snow | speckle_noise | contrast | elastic_transform | frost | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :--: | :-----------: | :--------: | :--------------: | :---------: | :------: | :--: | :-----------: | :------: | :---------------: | :---: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
|94.476| 94.420| 91.970| 93.644| 74.354| 87.238| 91.726| 93.064| 88.730| 86.000| 91.292| 90.646| 88.204| 76.294| 81.556| 88.744| 83.606| 92.286| 92.790|



