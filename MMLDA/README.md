


## MMLDA



> [Pang T., Du C., and Zhu J. Max-mahalanobis linear discriminant analysis networks. In International Conference on Machine Learning (ICML), 2018.](http://arxiv.org/abs/1802.09308)

> [Official-Code](https://github.com/P2333/Max-Mahalanobis-Training)

**Note:** The architectures used in MMLDA adopt auxiliary Linear layer after AveragePooling. We follows this design otherwise the training will not converge (unless a small learning rate given).

**Note:** The Conv module used in MMLDA adopts l2 regularizer with weight decay of 1e-4.

### CIFAR-10

    python train.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### CIFAR-100

    python train.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### MNIST

	python train.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### FashionMNIST

	python train.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10


## Evaluation

