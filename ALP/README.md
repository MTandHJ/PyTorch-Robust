

## ALP



> [Kannan H., Kurakin A. and Goodfellow I. Adversarial logit pairing. In Advances in Neural Information Processing Systems (NIPS), 2018.](https://arxiv.org/abs/1803.06373)



### CIFAR-10

    python ALP.py resnet32 cifar10 -lp=AT --epochs=200 -wd=0.0002 --leverage=0.5

### CIFAR-100

    python ALP.py resnet32 cifar10 -lp=AT --epochs=200 -wd=0.0002 --leverage=0.5

### MNIST

    python ALP.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=0.5

### FashionMNIST

    python ALP.py mnist fashiomnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=0.5



## Evaluation



### CIFAR10



| $\ell_{\infty}$ | leverage |   Net    |  LP  | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :-----------------------: | :------: | :------: | :--: | :----: | :---: | :----: | :----: | :------: | :---: |
|           8/255           |   0.5    | ResNet32 |  AT  | 80.70  | 53.18 | 47.76  | 46.95  |  49.54   | 43.23 |
|          16/255           |   0.5    | ResNet32 |  AT  | 80.70  | 33.85 | 18.83  | 16.55  |  27.22   | 12.27 |
|           8/255           |    1     | ResNet32 |  AT  | 77.12  | 54.04 | 49.34  | 48.61  |  49.53   | 44.89 |
|          16/255           |    1     | ResNet32 |  AT  | 77.12  | 36.26 | 22.35  | 19.64  |  28.56   | 14.34 |





| $\ell_2$ | leverage |   Net    |  LP  | Nat(%) | PGD-50 | DeepFool |  C&W  |  AA   |
| :---------------: | :------: | :------: | :--: | :----: | :----: | :------: | :---: | :---: |
|        0.5        |   0.5    | ResNet32 |  AT  | 80.70  | 56.22  |  58.91   | 54.26 | 53.34 |
|        0.5        |    1     | ResNet32 |  AT  | 77.12  | 56.34  |  58.00   | 53.91 | 53.25 |




| $\ell_1$ | leverage |   Net    |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :------: | :------: | :--: | :----: | :----: | :---: |
|        12         |   0.5    | ResNet32 |  AT  | 80.70  | 55.68  | 22.16 |
|        12         |    1     | ResNet32 |  AT  | 77.12  | 55.69  | 24.11 |



### MNIST



| $\ell_{\infty}$ | leverage |  Net  |  LP  | Nat(%) | FGSM  | PGD-50 | PGD-100 | DeepFool |  AA   |
| :-----------------------: | :------: | :---: | :--: | :----: | :---: | :----: | :-----: | :------: | :---: |
|            0.3            |   0.5    | mnist | null | 99.41  | 97.84 | 96.71  |  96.08  |  97.17   | 92.60 |



| $\ell_2|\epsilon$ | Leverage |  Net  |  LP  | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA   |
| :---------------: | :------: | :---: | :--: | :----: | :-----: | :------: | :---: | :---: |
|         2         |   0.5    | mnist | null | 99.41  |  95.72  |  97.62   | 89.07 | 16.90 |



| $\ell_1$ | leverage |  Net  |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :------: | :---: | :--: | :----: | :----: | :---: |
|        10         |   0.5    | mnist | null | 99.41  | 97.91  | 90.74 |



### FashionMNIST





| $\ell_{\infty}$ | leverage |  Net  |  LP  | Nat(%) | FGSM  | PGD-50 | PGD-100 | DeepFool |  AA   |
| :-----------------------: | :------: | :---: | :--: | :----: | :---: | :----: | :-----: | :------: | :---: |
|            0.3            |   0.5    | mnist | null | 83.08  | 68.21 | 61.40  |  55.82  |  56.35   | 24.25 |



| $\ell_2$ | Leverage |  Net  |  LP  | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA  |
| :---------------: | :------: | :---: | :--: | :----: | :-----: | :------: | :---: | :--: |
|         2         |   0.5    | mnist | null | 83.08  |  64.14  |  66.16   | 25.53 | 2.35 |



| $\ell_1$ | leverage |  Net  |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :------: | :---: | :--: | :----: | :----: | :---: |
|        10         |   0.5    | mnist | null | 83.08  | 65.74  | 53.65 |







