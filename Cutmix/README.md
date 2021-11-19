
## Cutmix


> [Yun S., Han D., Oh S., Chu S., Choe J. and Yoo Y. CutMix: Regularization strategy to train strong classifiers with localizable features. In International Conference on Computer Vision (ICCV), 2019.](https://arxiv.org/abs/1905.04899)

[official-code](https://github.com/clovaai/CutMix-PyTorch)


## Usage

**Note:** The follow config is for PyramidNet-200 but we employs ResNet18 here.

### CIFAR-10

	python STD.py resnet18 cifar10 --alpha=1 --cutmix-prob=0.5 -wd=1e-4 -lr=0.25 -lp=Cutmix --epochs=300 -b=64

### CIFAR-100

	python STD.py resnet18 cifar100 --alpha=1 --cutmix-prob=0.5 -wd=1e-4 -lr=0.25 -lp=Cutmix --epochs=300 -b=64



## Evaluation



### CIFAR-10



|      |   Net    |      | Clean |   C    |
| :--: | :------: | :--: | :---: | :----: |
|  STD  | ResNet18 | last | 96.18 | 73.643 |



| brightness | defocus_blur |  fog   | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate |  snow  | speckle_noise | contrast | elastic_transform | frost  | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :----: | :-----------: | :--------: | :--------------: | :---------: | :------: | :----: | :-----------: | :------: | :---------------: | :----: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
|94.798| 85.880| 90.236| 74.502| 55.860| 76.622| 82.594| 93.100| 86.552| 44.504| 84.380| 86.920| 80.336| 26.668| 51.814| 73.178| 39.334| 90.646| 81.292|

### CIFAR-100



|      |   Net    |      | Clean |   C    |
| :--: | :------: | :--: | :---: | :----: |
|  STD  | wrn_28_10 | last | 84.61 | 50.745 |



| brightness | defocus_blur |  fog   | gaussian_blur | glass_blur | jpeg_compression | motion_blur | saturate |  snow  | speckle_noise | contrast | elastic_transform | frost  | gaussian_noise | impulse_noise | pixelate | shot_noise | spatter | zoom_blur |
| :--------: | :----------: | :----: | :-----------: | :--------: | :--------------: | :---------: | :------: | :----: | :-----------: | :------: | :---------------: | :----: | :------------: | :-----------: | :------: | :--------: | :-----: | :-------: |
|79.314| 65.598| 71.620| 52.856| 20.910| 44.176| 60.338| 70.808| 63.344| 18.106| 64.850| 66.164| 53.494| 9.170| 21.346| 53.932| 15.792| 73.000| 59.342|


