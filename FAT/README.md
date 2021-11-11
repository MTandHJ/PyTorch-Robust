

## FAT

> [Zhang J., Xu X., Han B., Niu G., Cui L., Sugiyama M.,  Kankanhalli M. Attacks which do not kill training make adversarial learning stronger. In International Conference on Machine Learning (ICML), 2020.](http://arxiv.org/abs/2002.11242)

[official-code](https://github.com/zjfheart/Friendly-Adversarial-Training)



### Usage

#### CIFAR-10

	python AT.py resnet18 cifar10 --attack=fpgd-linf -lp=FAT
	python TRADES.py resnet18 cifar10 --attack=fpgd-linf -lp=FAT-TRADES



### Evaluation



### CIFAR-10



|           | $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------: | :-------------: | :------: | :--: | :---: | :---: |
|    AT     |      8/255      | ResNet18 | last | 87.34 | 41.56 |
|    AT     |      8/255      | ResNet18 | best | 87.55 | 44.10 |
| TRADES(6) |      8/255      | ResNet18 | last | 83.09 | 47.68 |
| TRADES(6) |      8/255      | ResNet18 | best | 82.71 | 48.16 |





