


## mixup

> [Zhang H., Cisse M., Dauphin Y. and Lopez-Paz D. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations (ICLR), 2018.](https://arxiv.org/abs/1710.09412)

[official-code](https://github.com/facebookresearch/mixup-cifar10)

## Usage

### CIFAR-10

	python STD.py resnet18 cifar10 --alpha=1 -wd=1e-4 --epochs=200 -lp=Rice2020ICML



## Evaluation



### CIFAR-10



|      |   Net    |      | Clean  |
| :--: | :------: | :--: | :----: |
|  AT  | ResNet18 | last | 95.860 |



| brightness | defocus_blur | fog  | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate | snow | speckle_noise | contrast | elastic_transform | frost | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :--: | :-----------: | :--------: | :--------------: | :---------: | :------: | :--: | :-----------: | :------: | :---------------: | :---: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
| 94.308| 87.570| 90.850| 78.532| 61.668| 81.818| 83.852| 91.844| 88.342| 69.618| 87.066| 87.378| 88.652| 58.446| 50.180| 81.420| 67.588| 87.976| 82.844 |

