


## AWP

> [Wu D., Xia S. and Wang Y. Adversarial weight perturbation helps robust generalization. In Advances in Neural Information Processing Systems (NIPS), 2020.](https://arxiv.org/pdf/2004.05884.pdf)

> [official-code](https://github.com/csdongxian/AWP)


## CIFAR-10


	python AT.py resnet18 cifar10 -lr=0.1 -lp=AWP --epochs=200 -wd=5e-4 --awp-gamma=0.01 --awp-warmup=0
	python TRADES.py resnet18 cifar10 -lr=0.1 -lp=AWP --epochs=200 -wd=5e-4 --awp-gamma=0.005 --awp-warmup=10

## CIFAR-100

	python AT.py resnet18 cifar100 -lr=0.1 -lp=AWP --epochs=200 -wd=5e-4 --awp-gamma=0.01 --awp-warmup=0
	python TRADES.py resnet18 cifar100 -lr=0.1 -lp=AWP --epochs=200 -wd=5e-4 --awp-gamma=0.005 --awp-warmup=10


