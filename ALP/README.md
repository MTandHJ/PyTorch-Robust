


## ALP



> [Kannan H., Kurakin A. and Goodfellow I. Adversarial logit pairing. In Advances in Neural Information Processing Systems (NIPS), 2018.](https://arxiv.org/abs/1803.06373)



### CIFAR-10

    python ALP.py resnet18 cifar10 -lp=AT --epochs=200 -wd=0.0002 --leverage=0.5

### CIFAR-100

    python ALP.py resnet18 cifar100 -lp=AT --epochs=200 -wd=0.0002 --leverage=0.5

### MNIST

    python ALP.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=0.5

### FashionMNIST

    python ALP.py mnist fashionmnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=0.5



## Evaluation



### CIFAR-10



| $\ell_{\infty}$ |   Net    | leverage |      | Clean |  AA   |
| :-------------: | :------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 |   0.5    | last | 85.70 | 41.20 |
|      8/255      | ResNet18 |   0.5    | best | 84.26 | 46.82 |







