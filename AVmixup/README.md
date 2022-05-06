


## AVmixup


> [Lee S., Lee H. and Yoon S. Adversarial vertex mixup: toward better adversarially robust generalization. In IEEE Conference on Computer Vsion and Pattern Recognition (CVPR), 2020.](https://arxiv.org/abs/2003.02484v3#:~:text=Adversarial%20Vertex%20mixup%20%28AVmixup%29%2C%20a%20soft-labeled%20data%20augmentation,and%20show%20that%20AVmixup%20significantly%20improves%20the%20robust)

[[official-code](https://github.com/xuyinhu/AVmixup)]
[[PyTorch-hirokiadachi-AVmixup](https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch)]



## Usage

### CIFAR-10

    python AT.py wrn_34_10 cifar10 --gamma=2 --lambda1=1 --lambda2=0.1

### CIFAR-100

    python AT.py wrn_34_10 cifar100 --gamma=2 --lambda1=1 --lambda2=0.1

### SVHN

    python AT.py wrn_34_10 svhn --gamma=2 --lambda1=1 --lambda2=0.1

### Tiny ImageNet

    python AT.py wrn_34_10 tinyimagenet --gamma=2 --lambda1=1 --lambda2=0.1


## Evaluation

**Note:** So far I haven't tested it on Wide ResNet, which is the default model used in the paper. 
Therefore, I am not sure if this implementation is correct.

### CIFAR-10


|    | $\ell_{\infty}$ |   Net    |      | Clean | PGD-10 |  AA   |
|:--:|:---------------:|:--------:|:----:|:-----:|:------:|:-----:|
| AT |      8/255      | ResNet18 | last | 76.92 | 53.57  |  --   |
| AT |      8/255      | ResNet18 | best | 89.07 | 61.30  | 20.57 |

