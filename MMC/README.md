


## MMC



> [Pang T., Xu K., Dong Y., Du C., Chen N. and Zhu J. Rethinking softmax cross-entropy loss for adversarial robustness. In International Conference on Learning Representations (ICLR), 2020.](http://arxiv.org/abs/1905.10626)

> [Official-Code](https://github.com/P2333/Max-Mahalanobis-Training)


**Note:** The architectures used in MMC adopt auxiliary Linear layer after AveragePooling. We follows this design otherwise the training will not converge (unless a small learning rate given).

**Note:** The Conv module used in MMC adopts l2 regularizer with weight decay of 1e-4.

### CIFAR-10

    python STD.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### CIFAR-100

    python STD.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### MNIST

	python STD.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### FashionMNIST

	python STD.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10


## Evaluation

