This repo is built for consistent comparisons of defense methods. However, I can not make sure these implementations are consistent with the official codes. You may find some tiny differences of the design logic for each defense method. Because I usually implement a new method based on the latest framework in [here]([MTandHJ/rfk: a simpe FrameworK in terms of Robustness (github.com)](https://github.com/MTandHJ/rfk)).



## Usage



```
┌── data # the path of data
│	├── mnist
│	├── fashionmnist
│	├── svhn
│	├── cifar10
│	└── cifar100
└── Pytorch-Robust
	└── Defense
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
        ├── transfer_attack.py #
        └── white_box_attack.py # the white-box attacks due to foolbox
```



You may need rewrite the config.py based on your environment.

