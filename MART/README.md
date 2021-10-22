

## MART


> Wang Y., Zou D., Yi J., Bailey J., Ma X., Gu Q. Improving adversarial robustness requires revisiting misclassified examples. In International Conference on Learning Representations (ICML), 2020.

[Official-Code](https://github.com/YisenWang/MART)



**Note:** The official-code adopts a absolute step size of 0.007 but we use 2/255 following PGD-AT. There exists difference of settings between the codes and those claimed in the paper. We follows the official code here.

 

### CIFAR-10

    python train.py resnet18 cifar10 -lp=MART --epochs=120 -lr=0.01 -wd=3.5e-3 --leverage=5
    python train.py wrn_28_10 cifar10 -lp=MART --epochs=90 -lr=0.1 -wd=7e-4 --leverage=6

### CIFAR-100

    python train.py resnet18 cifar100 -lp=MART --epochs=120 -lr=0.01 -wd=3.5e-3 --leverage=5
    python train.py wrn_28_10 cifar100 -lp=MART --epochs=90 -lr=0.1 -wd=7e-4 --leverage=6



## Evaluation

### CIFAR10



| $\ell_{\infty}$ | leverage |   Net    |  LP  | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA   |
| :-------------: | :------: | :------: | :--: | :----: | :---: | :----: | :----: | :------: | :---: |
|      8/255      |    5     | ResNet18 |  AT  | 81.72  | 59.33 | 53.89  | 52.97  |  53.95   | 47.76 |
|     16/255      |    5     | ResNet18 |  AT  | 81.72  | 42.93 | 26.09  | 22.67  |  33.45   | 13.23 |





| $\ell_2$ | leverage |   Net    |  LP  | Nat(%) | PGD-50 | DeepFool | C&W  |  AA   |
| :------: | :------: | :------: | :--: | :----: | :----: | :------: | :--: | :---: |
|   0.5    |    5     | ResNet18 |  AT  | 81.72  | 61.62  |  62.18   |      | 57.55 |




| $\ell_1$ | leverage |   Net    |  LP  | Nat(%) | PGD-50 | SLIDE |
| :------: | :------: | :------: | :--: | :----: | :----: | :---: |
|    12    |    5     | ResNet18 |  AT  | 81.72  | 61.36  | 29.80 |

