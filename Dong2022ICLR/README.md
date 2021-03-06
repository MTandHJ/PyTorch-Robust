


## Dong2022ICLR




> [Dong Y., Xu K., Yang X., Pang T., Deng Z., Su H. and Zhu J.](http://arxiv.org/abs/2106.01606)

> [official-code](https://github.com/dongyp13/memorization-AT)


## Usage

### CIFAR-10

    python AT.py resnet18 cifar10 --alpha=0.9 --start-es=90 --end-es=150 --reg-weight=300
    python TRADES.py resnet18 cifar10 --alpha=0.9 --start-es=90 --end-es=150 --reg-weight=300

### CIFAR-100

    python AT.py resnet18 cifar100 --alpha=0.9 --start-es=90 --end-es=150 --reg-weight=300
    python TRADES.py resnet18 cifar100 --alpha=0.9 --start-es=90 --end-es=150 --reg-weight=300




## Evaluation


### PGD-TE

| $\ell_{\infty}$ |   Net    |      | Clean |  PGD-10   |
| :-------------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 | last | 82.98 | 54.56 |
|      8/255      | ResNet18 | best | 82.93 | 55.66 |
