



## Usage



```
┌── data # the path of data
│	├── mnist
│	├── fashionmnist
│	├── svhn
│	├── cifar10
│	└── cifar100
└── rfk
	├── autoattack # AutoAttack
	├── infos # for saving trained model
	├── logs # for logging
	├── models # Architectures
	├── src
		├── attacks.py # 
		├── base.py # Coach, arranging the training procdure
		├── config.py # You can specify the ROOT as the path of training data.
		├── criteria.py # useful criteria of foolbox
		├── datasets.py # 
		├── dict2obj.py #
		├── loadopts.py # for loading basic configs
		├── loss_zoo.py # The implementations of loss function ...
		└── utils.py # other usful tools
	├── auto_attack.py # Croce F.
	├── requirements.txt # requiresments of packages
	├── transfer_attack.py #
	└── white_box_attack.py # the white-box attacks due to foolbox
```



### Training

#### CIFAR-10


    python STD.py resnet18 cifar10 -lp=STD --epochs=164 -wd=0.0002
    python AT.py resnet18 cifar10 -lp=AT --epochs=200 -wd=0.0002
    python ALP.py resnet18 cifar10 -lp=AT --epochs=200 -wd=0.0002 --leverage=0.5
    python TRADES.py resnet18 cifar10 -lp=TRADES --epochs=76 -wd=0.0002 --leverage=6

In particular,

    python STD.py wrn_28_10 cifar10 -lp=STD-wrn --epochs=200 -wd=5e-4

when training Wide ResNet.

Early stopping against over-fitting:

```
python AT.py resnet18 cifar10 -lp=Pang2021ICLR --epochs=110 -wd=0.0005
python TRADES.py resnet18 cifar10 -lp=Pang2021ICLR --epochs=110 -wd=0.0005
```



#### MNIST

```
python STD.py mnist mnist -lp=null --epochs=50 --optimizer=adam -lr=0.0001 -wd=0  -b=128
python AT.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01
python ALP.py mnist mnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01
python TRADES.py mnist mnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=1
```



#### fashionmnist



```
python STD.py mnist fashionmnist -lp=null --epochs=50 --optimizer=adam -lr=0.0001 -wd=0  -b=128
python AT.py mnist fashionmnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01
python ALP.py mnist fashionmnist -lp=null --epochs=84 -lr=0.0001 -wd=0 -mom=0 --optimizer=adam -b=50 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=0.5
python TRADES.py mnist fashionmnist -lp=TRADES-M --epochs=100 -lr=0.01 -wd=0 -mom=0.9 -b=128 --epsilon=0.3 --steps=40 --stepsize=0.01 --leverage=1
```





### Evaluation




Set the saved path as SP.

    python white_box_attack.py resnet18 cifar10 SP --attack=pgd-linf --epsilon_min=0 --epsilon_max=1 --epsilon_times=20
    python transfer_attack.py resnet18 SP1 resnet32 SP2 cifar10
    python auto_attack.py resnet18 cifar10 SP --norm=Linf --version=standard



**Note:** The stepsize in white_box_attack.py denotes the relative stepsize !



## Settings



### CIFAR

#### $\ell_{\infty}(\epsilon=8/255)$



|                  | FGSM | PGD-10 | PGD-20 | PGD-40 | DeepFool |  AA  |
| :--------------: | :--: | :----: | :----: | :----: | :------: | :--: |
|     stepsize     |  -   |  0.25  |  0.25  |  0.1   |   0.02   |  -   |
|      steps       |  -   |   10   |   20   |   40   |    50    |  -   |
|   rel_stepsize   |  -   |  0.25  |  0.1   |  0.1   |    -     |  -   |
|   abs_stepsize   |  -   | 2/255  | 2/255  | 0.0031 |    -     |  -   |
| initial_stepsize |  -   |   -    |   -    |   -    |    -     |  -   |
|    overshoot     |  -   |   -    |   -    |   -    |   0.02   |  -   |
|        lr        |  -   |   -    |   -    |   -    |    -     |  -   |



#### $\ell_2 (\epsilon=0.5)$



|                  | PGD-50 | DeepFool | C&W  |  AA  |
| :--------------: | :----: | :------: | :--: | :--: |
|     stepsize     |  0.1   |   0.02   | 0.01 |  -   |
|      steps       |   50   |    50    | 1000 |  -   |
|   rel_stepsize   |  0.1   |    -     |  -   |  -   |
|   abs_stepsize   |  0.05  |    -     |  -   |  -   |
| initial_stepsize |   -    |    -     |  -   |  -   |
|    overshoot     |   -    |   0.02   |  -   |  -   |
|        lr        |   -    |    -     | 0.01 |  -   |



#### $\ell_1 (\epsilon=12)$



|                  | PGD-50 | Sparse |
| :--------------: | :----: | :----: |
|     stepsize     |  0.05  |  0.05  |
|      steps       |   50   |   50   |
|   rel_stepsize   |  0.05  |  0.05  |
|   abs_stepsize   |  0.6   |  0.6   |
| initial_stepsize |   -    |   -    |
|    overshoot     |   -    |   -    |
|        lr        |   -    |   -    |



### MNIST

#### $\ell_{\infty} (\epsilon=0.3)$



|                  | FGSM |  PGD-50  | PGD-100  | DeepFool |  AA  |
| :--------------: | :--: | :------: | :------: | :------: | :--: |
|     stepsize     |  -   | 0.033333 | 0.033333 |   0.02   |  -   |
|      steps       |  -   |    50    |   100    |    50    |  -   |
|   rel_stepsize   |  -   | 0.033333 | 0.033333 |    -     |  -   |
|   abs_stepsize   |  -   |   0.01   |   0.01   |    -     |  -   |
| initial_stepsize |  -   |    -     |    -     |    -     |  -   |
|    overshoot     |  -   |    -     |    -     |   0.02   |  -   |
|        lr        |  -   |    -     |    -     |    -     |  -   |



#### $\ell_2 (\epsilon=2)$



|                  | PGD-100 | DeepFool | C&W  |  AA  |
| :--------------: | :-----: | :------: | :--: | :--: |
|     stepsize     |  0.05   |   0.02   | 0.01 |  -   |
|      steps       |   100   |    50    | 1000 |  -   |
|   rel_stepsize   |  0.05   |    -     |  -   |  -   |
|   abs_stepsize   |   0.1   |    -     |  -   |  -   |
| initial_stepsize |    -    |    -     |  -   |  -   |
|    overshoot     |    -    |   0.02   |  -   |  -   |
|        lr        |    -    |    -     | 0.01 |  -   |



#### $\ell_1 (\epsilon=10)$



|                  | PGD-50 | SLIDE |
| :--------------: | :----: | :---: |
|     stepsize     |  0.05  | 0.05  |
|      steps       |   50   |  50   |
|   rel_stepsize   |  0.05  | 0.05  |
|   abs_stepsize   |  0.5   |  0.5  |
| initial_stepsize |   -    |   -   |
|    overshoot     |   -    |   -   |
|        lr        |   -    |   -   |



## Results



### CIFAR-10

#### $\ell_{\infty}$



|        $\epsilon$        |     -     |    -    |   0    | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 | 8/255  | 16/255 |  8/255   |  16/255  | 8/255 | 16/255 | 8/255  | 16/255 |
| :----------------------: | :-------: | :-----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :------: | :------: | :---: | :----: | :----: | :----: |
|          Method          |    Net    |   LP    | TA(%)  | PGD-10 | PGD-10 | PGD-20 | PGD-20 | PGD-40 | PGD-40 |   AA   |   AA   | DeepFool | DeepFool |  BBA  |  BBA   |  FGSM  |  FGSM  |
|           STD            | ResNet32  |   STD   | 93.270 | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.019   |  0.000   |   -   |   -    | 21.800 | 14.150 |
|            AT            | ResNet32  |   AT    | 79.420 | 48.300 | 18.410 | 48.440 | 19.280 | 47.460 | 15.500 | 42.990 | 10.920 |  48.700  |  25.500  |   -   |   -    | 53.390 | 35.140 |
|   ALP  $(\lambda=0.5)$   | ResNet32  |   AT    | 80.700 | 47.760 | 18.830 | 48.270 | 19.770 | 46.950 | 16.550 | 43.230 | 12.270 |  49.540  |  27.220  |  --   |   -    | 53.180 | 33.850 |
|    ALP $(\lambda=1)$     | ResNet32  |   AT    | 77.120 | 49.340 | 22.350 | 49.820 | 21.200 | 48.610 | 19.640 | 44.890 | 14.340 |  49.530  |  28.560  |       |        | 54.040 | 36.260 |
|          TRADES          | ResNet32  | TRADES  | 74.040 | 45.590 | 20.420 | 45.900 | 21.060 | 45.110 | 19.000 | 40.780 | 14.490 |  44.730  |  23.800  |   -   |   -    | 48.740 | 29.520 |
| TRADES $(1 / \lambda=1)$ | ResNet32  | TRADES  | 82.220 | 39.370 | 10.090 | 39.930 | 10.700 | 38.300 | 8.470  | 34.620 | 6.150  |  41.830  |  17.290  |   -   |   -    | 45.840 | 23.610 |
|           STD            | ResNet18  |   STD   | 95.280 | 0.030  | 0.000  | 0.010  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  2.030   |  0.030   |   -   |   -    | 37.960 | 24.560 |
|            AT            | ResNet18  |   AT    | 84.780 | 44.660 | 16.649 | 45.450 | 17.530 | 43.210 | 13.670 | 41.400 | 8.490  |  50.230  |  26.670  |   -   |   -    | 53.400 | 35.060 |
|          TRADES          | ResNet18  | TREADES | 81.030 | 51.400 | 23.510 | 51.720 | 24.290 | 50.660 | 21.440 | 47.200 | 16.630 |  52.250  |  31.320  |   -   |   -    | 55.880 | 36.770 |
|           STD            | WRN_28_10 | STD-wrn | 96.070 | 0.230  | 0.080  | 0.070  | 0.030  | 0.020  | 0.000  | 0.000  | 0.000  |  9.630   |  1.220   |   -   |   -    | 48.950 | 32.820 |
|            AT            | WRN_28_10 |   AT    | 86.400 | 47.370 | 22.270 | 48.100 | 23.600 | 48.570 | 17.910 | 44.270 | 9.920  |  53.990  |  36.850  |   -   |   -    | 56.380 | 42.010 |
|          TRADES          | WRN_28_10 | TRADES  | 83.910 | 54.160 | 24.720 | 54.580 | 25.810 | 53.270 | 21.810 | 50.090 | 17.780 |  55.430  |  34.090  |   -   |   -    | 58.940 | 39.680 |
|           STD            |   cifar   |   STD   | 91.560 | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |  0.000   |  0.000   |   -   |   -    | 9.800  | 8.000  |
|            AT            |   ciar    |   AT    | 76.260 | 45.380 | 16.400 | 45.800 | 17.220 | 44.510 | 14.020 | 39.680 | 9.450  |  45.110  |  21.310  |   -   |   -    | 50.620 | 30.500 |
|          TRADES          |   cifar   | TRADES  | 72.510 | 42.040 | 17.610 | 42.220 | 18.180 | 41.610 | 16.490 | 36.890 | 11.600 |  40.610  |  20.190  |   -   |   -    | 45.220 | 25.660 |



#### $\ell_2$



| $\epsilon$ |    -     |    -    |   0    |  0.5   | 0.5  |  0.5  |   0.5    |   0.5   |
| :--------: | :------: | :----: | :----: | :--: | :---: | :------: | :------: | :------: |
|   Method   |   Net    |   LP   | TA(%)  | PGD-50 |  AA  |  C&W  | DeepFool | BBA |
|    STD     | ResNet32 | STD | 93.270 | 0.010 | 0.000 | 0.000 | 0.660 | - |
|     AT     | ResNet32 | AT | 79.420 | 56.700 | 53.310 |   54.670    |  58.480  |  -  |
| ALP $(\lambda=0.5)$ | ResNet32 | AT | 80.700 | 56.220 | 53.340 | 54.260 | 58.910 |  |
| ALP $(\lambda=1)$ | ResNet32 | AT | 77.120 | 56.340 | 53.250 | 53.910 | 58.000 | |
|   TRADES   | ResNet32 | TRADES | 74.040 | 54.290 | 51.000 | 51.320 |  54.850  | - |
| TRADES $(1 / \lambda=1)$ | ResNet32 | TRADES | 82.220 | 54.680 | 51.260 | 51.880 | 56.800 | - |
| STD | ResNet18 | STD | 95.280 | 0.340 | 0.000 | 0.030 | 13.860 | - |
| AT | ResNet18 | AT | 84.780 | 54.970 | 53.800 | 54.920 | 60.530 | - |
| TRADES | ResNet18 | TRADES | 81.030 | 59.900 | 56.850 | 57.420 |  62.040  | - |
| STD | WRN_28_10 | STD-wrn | 96.070 | 0.430 | 0.000 | 0.040 | 33.680 | - |
| AT | WRN_28_10 | AT | 86.400 | 53.950 | 52.680 | 53.600 | 62.600 | - |
| TRADES | WRN_28_10 | TRADES | 83.910 | 59.660 | 56.660 | 57.340 | 63.470 | - |
| STD | cifar | STD | 91.560 | 0.060 | 0.000 | 0.030 | 1.140 | - |
| AT | cifar | AT | 76.260 | 58.180 | 54.850 | 55.750 | 57.780 | - |
| TRADES | cifar | TRADES | 72.510 | 54.720 | 51.360 | 51.580 |  53.740  | - |



#### $\ell_1$



|        $\epsilon$        |     -     |    -    |   0    |   12   |   12   |
| :----------------------: | :-------: | :-----: | :----: | :----: | :----: |
|          Method          |    Net    |   LP    | TA(%)  | PGD-50 | SLIDE  |
|           STD            | ResNet32  |   STD   | 93.270 | 0.620  | 0.690  |
|            AT            | ResNet32  |   AT    | 79.420 | 57.250 | 23.080 |
|   ALP $(\lambda=0.5)$    | ResNet32  |   AT    | 80.700 | 55.680 | 22.160 |
|    ALP $(\lambda=1)$     | ResNet32  |   AT    | 77.120 | 55.690 | 24.110 |
|          TRADES          | ResNet32  | TRADES  | 74.040 | 54.950 | 25.240 |
| TRADES ($1 / \lambda=1$) | ResNet32  | TRADES  | 82.220 | 58.170 | 18.260 |
|           STD            | ResNet18  |   STD   | 95.280 | 6.600  | 3.210  |
|            AT            | ResNet18  |   AT    | 84.780 | 55.380 | 21.360 |
|          TRADES          | ResNet18  | TRADES  | 81.030 | 59.320 | 27.510 |
|           STD            | WRN_28_10 | STD-wrn | 96.070 | 7.890  | 6.520  |
|            AT            | WRN_28_10 |   AT    | 86.400 | 52.060 | 20.650 |
|          TRADES          | WRN_28_10 | TRADES  | 83.910 | 56.740 | 23.110 |
|           STD            |   cifar   |   STD   | 91.560 | 1.890  | 0.080  |
|            AT            |   cifar   |   AT    | 76.260 | 61.110 | 26.350 |
|          TRADES          |   cifar   | TRADES  | 72.510 | 57.610 | 26.560 |



### MNIST



#### $\ell_{\infty}$



|        $\epsilon$        |   -   |    -     |   0    |  0.3   |   0.3   |  0.3   |   0.3    | 0.3  |  0.3   |
| :----------------------: | :---: | :------: | :----: | :----: | :-----: | :----: | :------: | :--: | :----: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-50 | PGD-100 |   AA   | DeepFool | BBA  |  FGSM  |
|           STD            | mnist |   null   |        |        |         |        |          |  -   |        |
|            AT            | mnist |   null   | 99.460 | 96.270 | 95.480  | 92.780 |  96.890  |  -   | 97.500 |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 99.410 | 96.710 | 96.080  | 92.600 |  97.170  |  -   | 97.840 |
|    ALP $(\lambda=1)$     | mnist |   null   |        |        |         |        |          |  -   |        |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 99.450 | 96.220 | 95.560  | 92.990 |  96.900  |  -   | 97.590 |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 99.230 | 96.550 | 95.830  | 94.010 |  97.300  |  -   | 97.720 |



#### $\ell_2$




|        $\epsilon$        |   -   |    -     |   0    |    2    |   2    |    2     |   2    |  2   |
| :----------------------: | :---: | :------: | :----: | :-----: | :----: | :------: | :----: | :--: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-100 |   AA   | DeepFool |  C&W   | BBA  |
|           STD            | mnist |   null   |        |         |        |          |        |  -   |
|            AT            | mnist |   null   | 99.460 | 92.710  | 14.100 |  96.010  | 72.410 |  -   |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 99.410 | 95.720  | 16.900 |  97.620  | 89.070 |  -   |
|    ALP $(\lambda=1)$     | mnist |   null   |        |         |        |          |        |  -   |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 99.450 | 93.500  | 18.600 |  96.160  | 80.390 |  -   |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 99.230 | 96.830  | 9.160  |  96.890  | 90.610 |  -   |



#### $\ell_1$



|        $\epsilon$        |   -   |    -     |   0    |   10   |   10   |  10  |
| :----------------------: | :---: | :------: | :----: | :----: | :----: | :--: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-50 | SLIDE  | BBA  |
|           STD            | mnist |   null   |        |        |        |  -   |
|            AT            | mnist |   null   | 99.460 | 95.890 | 86.420 |  -   |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 99.410 | 97.910 | 90.740 |  -   |
|    ALP $(\lambda=1)$     | mnist |   null   |        |        |        |  -   |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 99.450 | 95.900 | 86.960 |  -   |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 99.230 | 96.950 | 90.360 |  -   |



## FashionMNIST



The same Setup as MNIST.



### $\ell_{\infty}$



|        $\epsilon$        |   -   |    -     |   0    |  0.3   |   0.3   |  0.3   |   0.3    | 0.3  |  0.3   |
| :----------------------: | :---: | :------: | :----: | :----: | :-----: | :----: | :------: | :--: | :----: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-50 | PGD-100 |   AA   | DeepFool | BBA  |  FGSM  |
|           STD            | mnist |   null   | 91.82  |  0.00  |  0.00   |  0.00  |   0.00   |  -   |  2.12  |
|            AT            | mnist |   null   | 77.760 | 61.970 | 56.870  | 45.990 |  64.040  |  -   | 70.550 |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 83.080 | 61.400 | 55.820  | 24.250 |  56.350  |  -   | 68.210 |
|    ALP $(\lambda=1)$     | mnist |   null   |        |        |         |        |          |  -   |        |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 86.060 | 58.070 | 51.170  | 29.610 |  56.130  |  -   | 67.890 |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 78.050 | 56.360 | 50.000  | 34.22  |  54.910  |  -   | 62.490 |



### $\ell_2$



|        $\epsilon$        |   -   |    -     |   0    |    2    |   2   |    2     |   2    |  2   |
| :----------------------: | :---: | :------: | :----: | :-----: | :---: | :------: | :----: | :--: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-100 |  AA   | DeepFool |  C&W   | BBA  |
|           STD            | mnist |   null   | 91.82  |  0.02   | 0.00  |   0.02   |  0.00  |  -   |
|            AT            | mnist |   null   | 77.760 | 62.190  | 0.190 |  65.180  | 48.090 |  -   |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 83.080 | 64.140  | 2.350 |  66.160  | 25.530 |  -   |
|    ALP $(\lambda=1)$     | mnist |   null   |        |         |       |          |        |  -   |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 86.060 | 65.200  | 3.460 |  65.850  | 26.610 |  -   |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 78.050 | 56.300  | 1.650 |  57.650  | 22.150 |  -   |





### $\ell_1$



|        $\epsilon$        |   -   |    -     |   0    |   10   |   10   |  10  |
| :----------------------: | :---: | :------: | :----: | :----: | :----: | :--: |
|          Method          |  Net  |    LP    | TA(%)  | PGD-50 | SLIDE  | BBA  |
|           STD            | mnist |   null   | 91.82  | 24.50  |  8.27  |  -   |
|            AT            | mnist |   null   | 77.760 | 67.200 | 57.990 |  -   |
|   ALP $(\lambda=0.5)$    | mnist |   null   | 83.080 | 65.740 | 53.650 |  -   |
|    ALP $(\lambda=1)$     | mnist |   null   |        |        |        |  -   |
| TRADES $(1 / \lambda=1)$ | mnist | TRADES-M | 86.060 | 68.880 | 58.400 |  -   |
| TRADES $(1/\lambda = 6)$ | mnist | TRADES-M | 78.050 | 63.980 | 50.340 |  -   |

