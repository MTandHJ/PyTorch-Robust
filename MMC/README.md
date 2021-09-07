


## PGD-AT



> [Madry A., Makelov A., Schmidt L., Tsipras D., Vladu A. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.](http://arxiv.org/abs/1706.06083)

> [CIFAR-10-Challenge](https://github.com/MadryLab/mnist_challenge)

### CIFAR-10

    python STD.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10
	python AT.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10

### CIFAR-100

    python STD.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10
	python AT.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10

### MNIST

	python STD.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10
	python AT.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10

### FashionMNIST

	python STD.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10
	python AT.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=0 -lr=0.01 --scale=10


## Evaluation

