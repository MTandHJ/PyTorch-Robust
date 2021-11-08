


## Rice2020ICML


> [Rice L., Wong E. \& Kolter J. Z. Overfitting in adversarially robust deep learning. In International Conference on Machine Learning (ICML), 2020.](http://arxiv.org/abs/2002.11569)

[official-code](https://github.com/locuslab/robust_overfitting)



### CIFAR-10

    python AT.py resnet18 cifar10 -lp=Rice2020ICML --epochs=200 -wd=5e-4 --ratio=0.02

### CIFAR-100


    python AT.py resnet18 cifar100 -lp=Rice2020ICML --epochs=200 -wd=5e-4 --ratio=0.02



## Evaluation



### CIFAR10

AT:

| $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 | last | 84.96 | 41.23 |
|      8/255      | ResNet18 | best | 83.25 | 47.63 |



ALP:

| $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 | last | 86.08 | 43.19 |
|      8/255      | ResNet18 | best | 85.98 | 46.25 |



TRADES:

| $\ell_{\infty}$ |   Net    |      | Clean |  AA   |
| :-------------: | :------: | :--: | :---: | :---: |
|      8/255      | ResNet18 | last | 83.78 | 47.30 |
|      8/255      | ResNet18 | best | 83.63 | 49.19 |



