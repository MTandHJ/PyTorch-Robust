


## AVmixup


> [Lee S., Lee H. and Yoon S. Adversarial vertex mixup: toward better adversarially robust generalization. In IEEE Conference on Computer Vsion and Pattern Recognition (CVPR), 2020.](https://arxiv.org/abs/2003.02484v3#:~:text=Adversarial%20Vertex%20mixup%20%28AVmixup%29%2C%20a%20soft-labeled%20data%20augmentation,and%20show%20that%20AVmixup%20significantly%20improves%20the%20robust)

[official-code](https://github.com/xuyinhu/AVmixup) [PyTorch-hirokiadachi-AVmixup](https://github.com/hirokiadachi/Adversarial-vertex-mixup-pytorch)



## Usage

### CIFAR-10

    python AT.py wrn_34_10 cifar10 --gamma=2 --lambda1=1 --lambda2=0.1

### CIFAR-100

    python AT.py wrn_34_10 cifar10 --gamma=2 --lambda1=1 --lambda2=0.1

### SVHN

    python AT.py wrn_34_10 svhn --gamma=2 --lambda1=1 --lambda2=0.1

### Tiny ImageNet

    python AT.py wrn_34_10 tinyimagenet --gamma=2 --lambda1=1 --lambda2=0.1
