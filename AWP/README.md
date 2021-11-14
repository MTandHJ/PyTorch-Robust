

## AWP

> [Wu D., Xia S. and Wang Y. Adversarial weight perturbation helps robust generalization. In Advances in Neural Information Processing Systems (NIPS), 2020.](https://arxiv.org/pdf/2004.05884.pdf)

[official-code](https://github.com/csdongxian/AWP)

## Usage



**Note:** I found AWP **sometimes** killed the training when employing ResNet18 (Using official code can not circumvent this problem). Interestingly, this phenomenon has never been observed on PreActResNet18 over several seeds.

### CIFAR-10

	python AT.py resnet18 cifar10 --awp-gamma=0.01 --awp-warmup=0
	python TRADES.py resnet18 cifar10 --awp-gamma=0.005 --awp-warmup=10

### CIFAR-100

	python AT.py resnet18 cifar100 --awp-gamma=0.01 --awp-warmup=0
	python TRADES.py resnet18 cifar100 --awp-gamma=0.005 --awp-warmup=10

### SVHN

	python AT.py resnet18 svhn --awp-gamma=0.01 --awp-warmup=5 -lr=0.01 --stepsize=0.00392156862745098



## Evaluation



### CIFAR-10







|           | $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------: | :-------------: | :------: | :--: | :---: | :---: |
|    AT     |      8/255      | ResNet18 | last | 82.32 | 49.52 |
|    AT     |      8/255      | ResNet18 | best | 82.32 | 49.88 |
| TRADES(6) |      8/255      | ResNet18 | last | 82.69 | 51.84 |
| TRADES(6) |      8/255      | ResNet18 | best | 82.75 | 51.86 |











