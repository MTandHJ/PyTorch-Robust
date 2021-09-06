

## PGD-AT



> [Madry A., Makelov A., Schmidt L., Tsipras D., Vladu A. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.](http://arxiv.org/abs/1706.06083)

> [CIFAR-10-Challenge](https://github.com/MadryLab/mnist_challenge)

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



| $\ell_{\infty}$ |    Net    |  LP  | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :-------------: | :-------: | :--: | :----: | :---: | :----: | :----: | :------: | :---: |
|      8/255      |   cifar   |  AT  | 76.26  | 50.62 | 45.38  | 45.51  |  45.11   | 39.68 |
|     16/255      |   cifar   |  AT  | 76.26  | 30.50 | 16.40  | 14.02  |  21.31   | 9.45  |
|      8/255      | ResNet32  |  AT  | 79.42  | 53.39 | 48.30  | 47.46  |  48.70   | 42.99 |
|     16/255      | ResNet32  |  AT  | 79.42  | 35.14 | 18.41  | 15.50  |  25.50   | 10.92 |
|      8/255      | ResNet18  |  AT  | 84.78  | 53.40 | 44.66  | 43.21  |  50.23   | 41.40 |
|     16/255      | ResNet18  |  AT  | 84.78  | 35.06 | 16.65  | 13.67  |  26.67   | 8.49  |
|      8/255      | WRN_28_10 |  AT  | 86.40  | 56.38 | 47.37  | 48.57  |  53.99   | 44.27 |
|     16/255      | WRN_28_10 |  AT  | 86.40  | 42.01 | 22.27  | 17.91  |  36.85   | 9.92  |





| $\ell_2$ |    Net    |  LP  | Nat(%) | PGD-50 | DeepFool |  C&W  |  AA   |
| :---------------: | :-------: | :--: | :----: | :----: | :------: | :---: | :---: |
|        0.5        |   cifar   |  AT  | 76.26  | 58.18  |  57.78   | 55.75 | 54.85 |
|        0.5        | ResNet32  |  AT  | 79.42  | 56.70  |  58.48   | 54.67 | 53.31 |
|        0.5        | ResNet18  |  AT  | 84.78  | 54.97  |  60.53   | 54.92 | 53.80 |
|        0.5        | WRN_28_10 |  AT  | 86.40  | 58.18  |  57.58   | 55.72 | 54.85 |



| $\ell_1$ |    Net    |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :-------: | :--: | :----: | :----: | :---: |
|        12         |   cifar   |  AT  | 76.26  | 61.11  | 26.35 |
|        12         | ResNet32  |  AT  | 79.42  | 57.25  | 23.08 |
|        12         | ResNet18  |  AT  | 84.78  | 55.38  | 21.36 |
|        12         | WRN_28_10 |  AT  | 86.40  | 52.06  | 20.65 |



### MNIST



| $\ell_{\infty}$ |  Net  |  LP  | Nat(%) | FGSM | PGD-50 | PGD-100 | DeepFool |  AA   |
| :-----------------------: | :---: | :--: | :----: | :--: | :----: | :-----: | :------: | :---: |
|            0.3            | mnist | null | 99.46  | 97.5 | 96.27  |  95.48  |  96.89   | 92.78 |



| $\ell_2$ |  Net  |  LP  | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA   |
| :------: | :---: | :--: | :----: | :-----: | :------: | :---: | :---: |
|    2     | mnist | null | 99.46  |  92.71  |  96.01   | 72.41 | 14.10 |



| $\ell_1$ |  Net  |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :---: | :--: | :----: | :----: | :---: |
|        10         | mnist | null | 99.46  | 95.89  | 86.42 |



### FashionMNIST





| $\ell_{\infty}$ |  Net  |  LP  | Nat(%) | FGSM  | PGD-50 | PGD-100 | DeepFool |  AA   |
| :-----------------------: | :---: | :--: | :----: | :---: | :----: | :-----: | :------: | :---: |
|            0.3            | mnist | null | 77.76  | 70.55 | 61.97  |  56.87  |  64.04   | 45.99 |



| $\ell_2$ |  Net  |  LP  | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA  |
| :---------------: | :---: | :--: | :----: | :-----: | :------: | :---: | :--: |
|         2         | mnist | null | 77.76  |  62.19  |  65.18   | 48.09 | 0.19 |



| $\ell_1$ |  Net  |  LP  | Nat(%) | PGD-50 | SLIDE |
| :---------------: | :---: | :--: | :----: | :----: | :---: |
|        10         | mnist | null | 77.76  | 67.20  | 57.99 |





