

## Pang2021ICLR




> [Pang T., Yang X., Dong Y., Su H. and Zhu J. Bag of tricks for adversarial training. In International Conference on Learning Representations.](http://arxiv.org/abs/2010.00467)

> [official-code](https://github.com/P2333/Bag-of-Tricks-for-AT)

We use the baseline settings:

Batch size 128; initial learning rate 0.1 (decay factor 10 at 100 and 105 epochs, totally 110 epochs);
SGD momentum optimizer; weight decay 5 × 10−4 ; eval mode BN for generating adversarial
examples.

### CIFAR-10

    python AT.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python ALP.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar10 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 --leverage=6

### CIFAR-100

    python AT.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python ALP.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 
    python TRADES.py resnet18 cifar100 -lr=0.1 -lp=Pang2021ICLR --epochs=110 -wd=5e-4 --leverage=6



## Evaluation



### CIFAR10



| $\ell_{\infty}$ | method | leverage |   Net    |   LP    | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :-------------: | :----: | :------: | :------: | :-----: | :----: | :---: | :----: | :----: | :------: | :---: |
|      8/255      | PGD-AT |    -     | ResNet18 | default | 83.89  | 58.67 | 53.28  | 52.20  |  54.23   | 48.40 |
|     16/255      | PGD-AT |    -     | ResNet18 | default | 83.89  | 38.74 | 21.10  | 17.86  |  29.57   | 13.21 |
|      8/255      | TRADES |    6     | ResNet18 | default | 82.35  | 57.06 | 52.98  | 52.39  |  53.63   | 48.70 |
|     16/255      | TRADES |    6     | ResNet18 | default | 82.35  | 38.61 | 25.05  | 22.59  |  32.77   | 17.33 |



| $\ell_2$ | method | leverage |   Net    |   LP    | Nat(%) | PGD-50 | DeepFool |  C&W  |  AA   |
| :------: | :----: | :------: | :------: | :-----: | :----: | :----: | :------: | :---: | :---: |
|   0.5    | PGD-AT |    -     | ResNet18 | default | 83.89  | 62.16  |  63.65   | 60.19 | 58.85 |
|   0.5    | TRADES |    6     | ResNet18 | default | 82.35  | 60.86  |  63.33   | 58.54 | 57.97 |




| $\ell_1$ | method | leverage |   Net    |   LP    | Nat(%) | PGD-50 | SLIDE |
| :------: | :----: | :------: | :------: | :-----: | :----: | :----: | :---: |
|    12    | PGD-AT |    -     | ResNet18 | default | 83.89  | 61.77  | 23.48 |
|    12    | TRADES |    1     | ResNet18 | default | 82.35  | 60.33  | 28.25 |



### CIFAR-100



| $\ell_{\infty}$ | method | leverage |   Net    |   LP    | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :-------------: | :----: | :------: | :------: | :-----: | :----: | :---: | :----: | :----: | :------: | :---: |
|      8/255      | PGD-AT |    -     | ResNet18 | default | 59.12  | 33.12 | 29.92  | 29.42  |  28.60   | 25.36 |
|     16/255      | PGD-AT |    -     | ResNet18 | default | 59.12  | 18.22 | 10.63  |  9.65  |  13.01   | 7.37  |
|      8/255      | TRADES |    6     | ResNet18 | default | 56.98  | 31.39 | 28.68  | 28.24  |  26.60   | 23.80 |
|     16/255      | TRADES |    6     | ResNet18 | default | 56.98  | 17.50 | 11.99  | 11.28  |  12.58   | 8.13  |





| $\ell_2$ | method | leverage |   Net    |   LP    | Nat(%) | PGD-50 | DeepFool |  C&W  |  AA   |
| :------: | :----: | :------: | :------: | :-----: | :----: | :----: | :------: | :---: | :---: |
|   0.5    | PGD-AT |    -     | ResNet18 | default | 59.12  | 38.67  |  37.79   | 36.37 | 34.82 |
|   0.5    | TRADES |    6     | ResNet18 | default | 56.98  | 37.04  |  35.94   | 33.50 | 32.64 |




| $\ell_1$ | method | leverage |   Net    |   LP    | Nat(%) | PGD-50 | SLIDE |
| :------: | :----: | :------: | :------: | :-----: | :----: | :----: | :---: |
|    12    | PGD-AT |    -     | ResNet18 | default | 59.12  | 39.58  | 14.15 |
|    12    | TRADES |    1     | ResNet18 | default | 56.98  | 38.06  | 16.53 |

