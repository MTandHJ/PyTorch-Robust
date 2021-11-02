

## FastAT (FGSM-RS)


> [Wong E., Rice L., Kolter J.Z. Fast is better than free: revisiting adversarial training. In International Conference on Learning Representations (ICLR), 2020.](http://arxiv.org/abs/2001.03994)

**Note:** This implementation excludes the apex module. In addition, there are some differences for training MNIST and FashionMNIST (the learning policy actually).

**Note:** FastAT (FGSM-RS) crafts the adversarial samples in training mode !!!

You may prefer the official codes:

> [Official-Code](https://github.com/locuslab/fast_adversarial)

### CIFAR-10

    python train.py preact18 cifar10 -lp=Fast --epochs=15 -b=128 -wd=5e-4 --lr-min=0 --lr-max=0.2

### CIFAR-100

    python train.py preact18 cifar100 -lp=Fast --epochs=15 -b=128 -wd=5e-4 --lr-min=0 --lr-max=0.2

### MNIST

    python train.py mnist mnist -lp=Fast --epochs=10 -b=100 --optimizer=adam -wd=0 --epsilon=0.3 --stepsize=1.25 --lr_max=5e-3

### FashionMNIST

    python train.py mnist mnist -lp=Fast --epochs=10 -b=100 --optimizer=adam -wd=0 --epsilon=0.3 --stepsize=1.25 --lr_max=5e-3



## Evaluation



| $\epsilon$ |   Net    |  LP  | Nat(%) | PGD-40 |  AA   |
| :--------: | :------: | :--: | :----: | :----: | :---: |
|   8/255    | resnet18 | Fast | 77.70  | 45.62  | 41.35 |

