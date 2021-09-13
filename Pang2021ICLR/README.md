

## Pang2021ICLR


> [Pang T., Yang X., Dong Y., Su H. and Zhu J. Bag of tricks for adversarial training. In International Conference on Learning Representations.](http://arxiv.org/abs/2010.00467)

> [official-code](https://github.com/P2333/Bag-of-Tricks-for-AT)

We use the baseline settings:

Batch size 128; initial learning rate 0.1 (decay factor 10 at 100 and 105 epochs, totally 110 epochs);
SGD momentum optimizer; weight decay 5 × 10−4 ; eval mode BN for generating adversarial
examples.

### CIFAR-10

    python AT.py resnet18 cifar10 -lr=0.1 -lp=default --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar10 -lr=0.1 -lp=default --epochs=110 -wd=5e-4 --leverage=6

### CIFAR-100

    python AT.py resnet18 cifar100 -lr=0.1 -lp=default --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar100 -lr=0.1 -lp=default --epochs=110 -wd=5e-4 --leverage=6



## Evaluation



### CIFAR10


