


## Rice2020ICML


> [Rice L., Wong E. \& Kolter J. Z. Overfitting in adversarially robust deep learning. In International Conference on Machine Learning (ICML), 2020.](http://arxiv.org/abs/2002.11569)

[official-code](https://github.com/locuslab/robust_overfitting)



### CIFAR-10

    python AT.py resnet18 cifar10 -lp=Rice2020ICML --epochs=200 -wd=5e-4 --ratio=0.02

### CIFAR-100


    python AT.py resnet18 cifar100 -lp=Rice2020ICML --epochs=200 -wd=5e-4 --ratio=0.02



## Evaluation



We report the best checkpoint on the validation set.



| $\epsilon$ |   Net    |      LP      | Nat(%) | PGD-40 |  AA   |
| :--------: | :------: | :----------: | :----: | :----: | :---: |
|   8/255    | resnet18 | Rice2020ICML | 82.09  | 52.05  | 47.86 |

