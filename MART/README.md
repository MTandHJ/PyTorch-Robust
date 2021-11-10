

## MART


> Wang Y., Zou D., Yi J., Bailey J., Ma X., Gu Q. Improving adversarial robustness requires revisiting misclassified examples. In International Conference on Learning Representations (ICML), 2020.

[Official-Code](https://github.com/YisenWang/MART)



**Note:** There exists differences between the codes and those claimed in the paper. We follows the official code here.

 

### CIFAR-10

    python train.py resnet18 cifar10 -lp=MART --epochs=120 -lr=0.01 -wd=3.5e-3 --leverage=5
    python train.py wrn_28_10 cifar10 -lp=MART --epochs=90 -lr=0.1 -wd=7e-4 --leverage=6

### CIFAR-100

    python train.py resnet18 cifar100 -lp=MART --epochs=120 -lr=0.01 -wd=3.5e-3 --leverage=5
    python train.py wrn_28_10 cifar100 -lp=MART --epochs=90 -lr=0.1 -wd=7e-4 --leverage=6



## Evaluation



### CIFAR-10



| $\ell_{\infty}$ |   Net    | leverage |      | Clean |  AA   |
| :-------------: | :------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 |    5     | last | 82.77 | 47.29 |
|      8/255      | ResNet18 |    5     | best | 79.23 | 47.83 |





