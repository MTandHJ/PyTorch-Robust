


## MMC



> [Pang T., Xu K., Dong Y., Du C., Chen N. and Zhu J. Rethinking softmax cross-entropy loss for adversarial robustness. In International Conference on Learning Representations (ICLR), 2020.](http://arxiv.org/abs/1905.10626)

> [Official-Code](https://github.com/P2333/Max-Mahalanobis-Training)


**Note:** The architectures used in MMC adopt auxiliary Linear layer after AveragePooling. We follows this design otherwise the training will not converge (unless a small learning rate given).

**Note:** The Conv module used in MMC adopts l2 regularizer with weight decay of 1e-4.

### CIFAR-10

    python STD.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
    python AT.py resnet32 cifar10 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### CIFAR-100

    python STD.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
    python AT.py resnet32 cifar100 --epochs=200 -lp=MMC-C -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### MNIST

	python STD.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py mnist mnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10

### FashionMNIST

	python STD.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10
	python AT.py mnist fashionmnist --epochs=50 -lp=MMC-M -b=50 --optimizer=sgd -wd=1e-4 -lr=0.01 --scale=10


## Evaluation



### CIFAR-10



| $\ell_{\infty}$ |      |   Net    |  LP   | Nat(%) | FGSM  | PGD-10 | PGD-40 | DeepFool |  AA  |
| :-------------: | :--: | :------: | :---: | :----: | :---: | :----: | :----: | :------: | :--: |
|      8/255      | STD  | ResNet32 | MMC-C | 90.18  | 86.33 | 84.71  | 84.31  |  34.35   | 0.00 |
|     16/255      | STD  | ResNet32 | MMC-C | 90.18  | 86.65 | 78.22  | 77.63  |  12.50   | 0.00 |




| $\ell_2$ |      |   Net    |  LP   | Nat(%) | PGD-50 | DeepFool | C&W  |  AA  |
| :------: | :--: | :------: | :---: | :----: | :----: | :------: | :--: | :--: |
|   0.5    | STD  | ResNet32 | MMC-C | 90.18  | 88.02  |  53.38   | 2.41 | 0.09 |
|          |      |          |       |        |        |          |      |      |




| $\ell_1$ |      |   Net    |  LP   | Nat(%) | PGD-50 | SLIDE |
| :------: | :--: | :------: | :---: | :----: | :----: | :---: |
|    12    | STD  | ResNet32 | MMC-C | 90.18  | 88.30  | 86.80 |
|          |      |          |       |        |        |       |







