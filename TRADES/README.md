

## TRADES





> [Zhang H., Yu Y., Jiao J., Xing E., Ghaoui L. and Jordan M. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning (ICML), 2019.](https://arxiv.org/abs/1901.08573)

> [Official-Code](https://github.com/yaodongyu/TRADES)



**Note:** The official-code adopts a absolute step size of 0.007 but we use 2/255 following PGD-AT.



### CIFAR-10

    python train.py resnet18 cifar10 -lp=TRADES --epochs=76 -wd=0.0002 --leverage=6

### CIFAR-100

    python train.py resnet18 cifar100 -lp=TRADES --epochs=76 -wd=0.0002 --leverage=6

### MNIST

    python train.py mnist mnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=1

### FashionMNIST

    python train.py mnist fashionmnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=1



## Evaluation
