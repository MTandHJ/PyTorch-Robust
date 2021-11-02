
## FAT


> [Zhang J., Xu X., Han B., Niu G., Cui L., Sugiyama M. and  Kankanhalli M. Attacks which do not kill training make adversarial learning stronger. In International Conference on Machine Learning (ICML), 2020.](http://arxiv.org/abs/2002.11242)

[official-code](https://github.com/zjfheart/Friendly-Adversarial-Training)

### CIFAR-10

    python AT.py resnet18 cifar10 --tau=0 --omega=0.001 --random-type=uniform -lp=FAT-AT --epochs=120
	python TRADES.py resnet18 cifar10 --tau=0 --omega=0.0 --random-type=normal -lp=FAT-TRADES --epochs=85

### CIFAR-100

    python AT.py resnet18 cifar100 --tau=0 --omega=0.001 --random-type=uniform -lp=FAT-AT --epochs=120
	python TRADES.py resnet18 cifar100 --tau=0 --omega=0.0 --random-type=normal -lp=FAT-TRADES --epochs=85



## Evaluation