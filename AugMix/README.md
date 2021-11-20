


## AugMix

> [Hendrycks D., Norman M., Ekin D. C., Barret Z., Justin G. and Balaji L. AugMix: A simple data processing method to improve robustness and uncertainty. In International Conference on Learning Representations (ICLR), 2020.](https://arxiv.org/pdf/1912.02781.pdf)

[official-code](https://github.com/google-research/augmix)

## Usage

### CIFAR-10

	python STD.py resnet18 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 --jsd -wd=5e-4 --epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar10 --width=3 --depth=1 --severity=3 --alpha=1 --jsd -wd=5e-4 --epochs=100 -lp=AugMix

### CIFAR-100

	python STD.py resnet18 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --jsd --epochs=200 -lp=AugMix
	python STD.py wrn_28_10 cifar100 --width=3 --depth=1 --severity=3 --alpha=1 -wd=5e-4 --jsd --epochs=100 -lp=AugMix



## Evaluation



### CIFAR-10



|      |   Net    |      | Clean  |   C    |
| :--: | :------: | :--: | :----: | :----: |
|  STD  | ResNet18 | last | 95.690 | 88.475 |
|  STD( no jsd)  | ResNet18 | last | 95.500 | 86.537 |



| brightness | defocus_blur | fog  | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate | snow | speckle_noise | contrast | elastic_transform | frost | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :--: | :-----------: | :--------: | :--------------: | :---------: | :------: | :--: | :-----------: | :------: | :---------------: | :---: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
|94.476| 94.420| 91.970| 93.644| 74.354| 87.238| 91.726| 93.064| 88.730| 86.000| 91.292| 90.646| 88.204| 76.294| 81.556| 88.744| 83.606| 92.286| 92.790|
|94.596| 93.640| 91.552| 92.286| 68.756| 85.838| 90.220| 93.262| 87.800| 82.520| 88.782| 89.580| 86.670| 70.762| 80.030| 84.988| 79.786| 91.130| 92.016|


### CIFAR-100



|      |   Net    |      | Clean  |   C    |
| :--: | :------: | :--: | :----: | :----: |
|  STD  | wrn_28_10 | last | 81.01 | 67.418 |



| brightness | defocus_blur | fog  | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate | snow | speckle_noise | contrast | elastic_transform | frost | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :--: | :-----------: | :--------: | :--------------: | :---------: | :------: | :--: | :-----------: | :------: | :---------------: | :---: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
|78.080| 78.620| 71.482| 77.340| 45.594| 62.200| 73.834| 69.834| 68.606| 59.838| 73.854| 71.132| 66.006| 46.580| 59.516| 70.098| 56.768| 75.542| 76.034|


