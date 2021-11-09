

## PGD-AT



> [Madry A., Makelov A., Schmidt L., Tsipras D., Vladu A. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.](http://arxiv.org/abs/1706.06083)

> [CIFAR-10-Challenge](https://github.com/MadryLab/mnist_challenge)

### CIFAR-10

    python AT.py resnet18 cifar10 --epochs=200 -lp=AT -b=128 -wd=2e-4 

### CIFAR-100

    python AT.py resnet18 cifar100 --epochs=200 -lp=AT -b=128 -wd=2e-4 

### MNIST

    python AT.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01

### FashionMNIST

    python AT.py mnist fashionmnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01



## Latest Version



| $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 | last | 84.74 | 41.20 |
|      8/255      | ResNet18 | best | 84.26 | 46.82 |








