

## FGSM-AT



> [Goodfellow I.J., Shlens J., Szegedy C. Explaining and harnessing adversarial examples. In Conference On Learning Representations (ICLR 2015).](http://arxiv.org/abs/1412.6572)


### CIFAR-10

    python train.py resnet18 cifar10 --epochs=200 -lp=AT -b=128 -wd=2e-4 

### CIFAR-100

    python train.py resnet18 cifar100 --epochs=200 -lp=AT -b=128 -wd=2e-4 

### MNIST

    python train.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.0333333

### FashionMNIST

    python train.py mnist fashionmnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.0333333



## Evaluation





### CIFAR10



| $\ell_{\infty}$ |   Net    |  LP  | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA  |
| :-------------: | :------: | :--: | :----: | :---: | :----: | :----: | :------: | :--: |
|      8/255      | ResNet32 |  AT  | 88.23  | 54.75 |  0.04  |  0.01  |   0.17   | 0.00 |
|     16/255      | ResNet32 |  AT  | 88.23  | 33.68 |  0.01  |  0.00  |   0.00   | 0.00 |



| $\ell_2$ |   Net    |  LP  | Nat(%) | PGD-50 | DeepFool | C&W  |  AA  |
| :------: | :------: | :--: | :----: | :----: | :------: | :--: | :--: |
|   0.5    | ResNet32 |  AT  | 88.23  |  0.01  |   1.14   |      | 0.00 |




| $\ell_1$ | leverage |   Net    |  LP  | Nat(%) | PGD-50 | SLIDE |
| :------: | :------: | :------: | :--: | :----: | :----: | :---: |
|    12    |   0.5    | ResNet32 |  AT  | 88.23  |  2.08  | 29.54 |



### FashoinMNIST



