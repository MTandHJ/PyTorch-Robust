

## TRADES





> [Zhang H., Yu Y., Jiao J., Xing E., Ghaoui L. and Jordan M. Theoretically principled trade-off between robustness and accuracy. In International Conference on Machine Learning (ICML), 2019.](https://arxiv.org/abs/1901.08573)

> [Official-Code](https://github.com/yaodongyu/TRADES)

### CIFAR-10

    python train.py resnet18 cifar10 -lp=TRADES --epochs=76 -wd=0.0002 --leverage=6

### CIFAR-100

    python train.py resnet18 cifar100 -lp=TRADES --epochs=76 -wd=0.0002 --leverage=6

### MNIST

    python train.py mnist mnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=1

### FashionMNIST

    python train.py mnist fashionmnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.0333333 --leverage=1



## Evaluation



### CIFAR10



| $\ell_{\infty} $ | Leverage |    Net    |   LP   | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :--------------: | :------: | :-------: | :----: | :----: | :---: | :----: | :----: | :------: | :---: |
|      8/255       |    6     |   cifar   | TRADES | 72.51  | 45.22 | 42.04  | 41.61  |  40.61   | 36.89 |
|      16/255      |    6     |   cifar   | TRADES | 72.51  | 25.66 | 17.61  | 16.49  |  20.19   | 11.60 |
|      8/255       |    1     | ResNet32  | TRADES | 82.22  | 45.84 | 39.37  | 38.30  |  41.83   | 34.62 |
|      16/255      |    1     | ResNet32  | TRADES | 82.22  | 23.61 | 10.09  |  8.47  |  17.29   | 6.15  |
|      8/255       |    6     | ResNet32  | TRADES | 74.04  | 48.74 | 45.59  | 45.11  |  44.73   | 40.78 |
|      16/255      |    6     | ResNet32  | TRADES | 74.04  | 29.52 | 20.42  | 19.00  |  23.80   | 14.49 |
|      8/255       |    6     | ResNet18  | TRADES | 81.03  | 55.88 | 51.40  | 50.66  |  52.25   | 47.20 |
|      16/255      |    6     | ResNet18  | TRADES | 81.03  | 36.77 | 23.51  | 21.44  |  31.32   | 16.63 |
|      8/255       |    6     | WRN_28_10 | TRADES | 83.91  | 58.94 | 54.16  | 53.27  |  55.43   | 50.09 |
|      16/255      |    6     | WRN_28_10 | TRADES | 83.91  | 39.68 | 24.72  | 21.81  |  34.09   | 17.78 |





| $\ell_2$ | leverage |    Net    |   LP   | Nat(%) | PGD-50 | DeepFool |  C&W  |  AA   |
| :------: | :------: | :-------: | :----: | :----: | :----: | :------: | :---: | :---: |
|   0.5    |    6     |   cifar   | TRADES | 72.51  | 54.72  |  53.74   | 51.58 | 51.36 |
|   0.5    |    1     | ResNet32  | TRADES | 82.22  | 54.68  |  56.80   | 51.88 | 51.26 |
|   0.5    |    6     | ResNet32  | TRADES | 74.04  | 54.29  |  54.85   | 51.32 | 51.00 |
|   0.5    |    6     | ResNet18  | TRADES | 81.03  | 59.90  |  62.04   | 57.42 | 56.85 |
|   0.5    |    6     | WRN_28_10 | TRADES | 83.91  | 59.66  |  63.47   | 57.34 | 56.66 |




| $\ell_1$ | leverage |    Net    |   LP   | Nat(%) | PGD-50 | SLIDE |
| :------: | :------: | :-------: | :----: | :----: | :----: | :---: |
|    12    |    6     |   cifar   | TRADES | 72.51  | 57.61  | 26.56 |
|    12    |    1     | ResNet32  | TRADES | 82.22  | 58.17  | 18.26 |
|    12    |    6     | ResNet32  | TRADES | 74.04  | 54.95  | 25.24 |
|    12    |    6     | ResNet18  | TRADES | 81.03  | 59.32  | 27.51 |
|    12    |    6     | WRN_28_10 | TRADES | 83.91  | 56.74  | 23.11 |




### MNIST



| $\ell_{\infty} $ | leverage |  Net  |    LP    | Nat(%) | FGSM  | PGD-50 | PGD-100 | DeepFool |  AA   |
| :--------------: | :------: | :---: | :------: | :----: | :---: | :----: | :-----: | :------: | :---: |
|       0.3        |    1     | mnist | TRADES-M | 99.45  | 97.59 | 96.22  |  95.56  |  96.90   | 92.99 |
|       0.3        |    6     | mnist | TRADES-M | 99.23  | 97.72 | 96.55  |  95.83  |  97.30   | 94.01 |



| $\ell_2$ | Leverage |  Net  |    LP    | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA   |
| :------: | :------: | :---: | :------: | :----: | :-----: | :------: | :---: | :---: |
|    2     |   0.5    | mnist | TRADES-M | 99.45  |  93.50  |  96.16   | 80.39 | 18.60 |
|    2     |   0.5    | mnist | TRADES-M | 99.23  |  96.83  |  96.89   | 90.61 | 9.16  |



| $\ell_1$ | leverage |  Net  |    LP    | Nat(%) | PGD-50 | SLIDE |
| :------: | :------: | :---: | :------: | :----: | :----: | :---: |
|    10    |   0.5    | mnist | TRADES-M | 99.45  | 95.90  | 86.96 |
|    10    |   0.5    | mnist | TRADES-M | 99.23  | 96.95  | 90.36 |



### FashionMNIST



| $\ell_{\infty} $ | leverage |  Net  |    LP    | Nat(%) | FGSM  | PGD-50 | PGD-100 | DeepFool |  AA   |
| :--------------: | :------: | :---: | :------: | :----: | :---: | :----: | :-----: | :------: | :---: |
|       0.3        |    1     | mnist | TRADES-M | 86.06  | 67.89 | 58.07  |  51.17  |  56.13   | 29.61 |
|       0.3        |    6     | mnist | TRADES-M | 78.05  | 62.49 | 56.36  |  50.00  |  54.91   | 34.22 |



| $\ell_2$ | Leverage |  Net  |    LP    | Nat(%) | PGD-100 | DeepFool |  C&W  |  AA  |
| :------: | :------: | :---: | :------: | :----: | :-----: | :------: | :---: | :--: |
|    2     |   0.5    | mnist | TRADES-M | 86.06  |  65.20  |  65.85   | 26.61 | 3.46 |
|    2     |   0.5    | mnist | TRADES-M | 78.05  |  56.30  |  57.65   | 22.15 | 1.65 |



| $\ell_1$ | leverage |  Net  |    LP    | Nat(%) | PGD-50 | SLIDE |
| :------: | :------: | :---: | :------: | :----: | :----: | :---: |
|    10    |   0.5    | mnist | TRADES-M | 86.06  | 68.88  | 58.40 |
|    10    |   0.5    | mnist | TRADES-M | 78.05  | 63.98  | 50.34 |

