

## Pang2021ICLR




> [Pang T., Yang X., Dong Y., Su H. and Zhu J. Bag of tricks for adversarial training. In International Conference on Learning Representations.](http://arxiv.org/abs/2010.00467)

> [official-code](https://github.com/P2333/Bag-of-Tricks-for-AT)

We use the baseline settings:

Batch size 128; initial learning rate 0.1 (decay factor 10 at 100 and 105 epochs, totally 110 epochs);
SGD momentum optimizer; weight decay 5 × 10−4 ; eval mode BN for generating adversarial
examples.



## Usage

### CIFAR-10

    python AT.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python ALP.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 --leverage=6

### CIFAR-100

    python AT.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python ALP.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 --leverage=6



## Evaluation



### CIFAR-10





|           | $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------: | :-------------: | :------: | :--: | :---: | :---: |
|    AT     |      8/255      | ResNet32 | last | 76.65 | 43.87 |
|    AT     |      8/255      | ResNet32 | best | 76.60 | 43.88 |
|    AT     |      8/255      | ResNet18 | last | 84.15 | 48.72 |
|    AT     |      8/255      | ResNet18 | best | 84.15 | 48.72 |
| ALP(0.5)  |      8/255      | ResNet32 | last | 81.38 | 41.46 |
| ALP(0.5)  |      8/255      | ResNet32 | best | 81.36 | 41.53 |
| ALP(0.5)  |      8/255      | ResNet18 | last | 86.53 | 47.38 |
| ALP(0.5)  |      8/255      | ResNet18 | best | 86.53 | 47.38 |
| TRADES(6) |      8/255      | ResNet32 | last | 76.46 | 43.06 |
| TRADES(6) |      8/255      | ResNet32 | best | 76.63 | 43.25 |
| TRADES(6) |      8/255      | ResNet18 | last | 82.62 | 49.47 |
| TRADES(6) |      8/255      | ResNet18 | best | 82.35 | 49.54 |







