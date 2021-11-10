







# Here are some basic settings.
# It could be overwritten if you want to specify
# some configs. However, please check the correspoding
# codes in loadopts.py.



import torch
import logging
from .dict2obj import Config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = "../../data" # the path saving the data
DOWNLOAD = False # whether to download the data

SAVED_FILENAME = "paras.pt" # the filename of saved model paramters
PRE_BESTNAT = "nat"
PRE_BESTROB = "rob"
INFO_PATH = "./infos/{method}/{dataset}-{model}/{description}"
LOG_PATH = "./logs/{method}/{dataset}-{model}/{description}-{time}"
TIMEFMT = "%m%d%H"

# logger
LOGGER = Config(
    name='RFK', filename='log.txt', level=logging.DEBUG,
    filelevel=logging.DEBUG, consolelevel=logging.INFO,
    formatter=Config(
        filehandler=logging.Formatter('%(asctime)s:\t%(message)s'),
        consolehandler=logging.Formatter('%(message)s')
    )
)

# the seed for validloader preparation
VALIDSEED = 1

# default transforms
TRANSFORMS = {
    'mnist': 'tensor,none',
    'fashionmnist': 'tensor,none',
    'svhn': 'tensor,none',
    'cifar10': 'cifar,none',
    'cifar100': 'cifar,none',
    'validation': 'tensor,none'
}

VALIDER = {
    "mnist": Config(attack_type="pgd-linf", epsilon=0.3, stepsize=0.01, steps=100),
    "fashionmnist": Config(attack_type="pgd-linf", epsilon=0.3, stepsize=0.01, steps=100),
    "svhn": Config(attack_type="pgd-linf", epsilon=8/255, stepsize=2/255, steps=10),
    "cifar10": Config(attack_type="pgd-linf", epsilon=8/255, stepsize=2/255, steps=10),
    "cifar100": Config(attack_type="pgd-linf", epsilon=8/255, stepsize=2/255, steps=10),
}

# env settings
NUM_WORKERS = 3
PIN_MEMORY = True

# basic properties of inputs
BOUNDS = (0, 1) # for attacks
PREPROCESSING = None # for fb.attacks.Attack
MEANS = {
    "mnist": [0,],
    "fashionmnist": [0,],
    'svhn': [0.5, 0.5, 0.5],
    "cifar10": [0.4914, 0.4824, 0.4467],
    "cifar100": [0.5071, 0.4867, 0.4408]
}

STDS = {
    "mnist": [1,],
    "fashionmnist": [1,],
    'svhn': [0.5, 0.5, 0.5],
    "cifar10": [0.2471, 0.2435, 0.2617],
    "cifar100": [0.2675, 0.2565, 0.2761]
}

# the settings of optimizers of which lr could be pointed
# additionally.
OPTIMS = {
    "sgd": Config(lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=False, prefix="SGD:"),
    "adam": Config(lr=0.01, betas=(0.9, 0.999), weight_decay=0., prefix="Adam:")
}


# the learning schedule can be added here
LEARNING_POLICY = {
    "null": (
        "StepLR",
        Config(
            step_size=9999999999999,
            gamma=1,
            prefix="Null leaning policy will be applied:"
        )
    ),
   "Pang2021ICLR": (
        "MultiStepLR",
        Config(
            milestones=[100, 105],
            gamma=0.1,
            prefix="Pang2020ICLR leaning policy will be applied:"
        )
    ),
    "Rice2020ICML": (
        "MultiStepLR",
        Config(
            milestones=[100, 150],
            gamma=0.1,
            prefix="Rice2020ICML leaning policy will be applied:"
        )
    ),
    "STD": (
        "MultiStepLR",
        Config(
            milestones=[82, 123],
            gamma=0.1,
            prefix="STD leaning policy will be applied:"
        )
    ),
    "STD-wrn": (
        "MultiStepLR",
        Config(
            milestones=[60, 120, 160],
            gamma=0.2,
            prefix="STD-wrn leaning policy will be applied:"
        )
    ),
    "AT":(
        "MultiStepLR",
        Config(
            milestones=[102, 154],
            gamma=0.1,
            prefix="AT learning policy, an official config:"
        )
    ),
    "TRADES":(
        "MultiStepLR",
        Config(
            milestones=[75, 90, 100],
            gamma=0.1,
            prefix="TRADES learning policy, an official config:"
        )
    ),
    "TRADES-M":(
        "MultiStepLR",
        Config(
            milestones=[55, 75, 90],
            gamma=0.1,
            prefix="TRADES-M learning policy, an official config for MNIST:"
        )
    ),
    "cosine":(   
        "CosineAnnealingLR",   
        Config(          
            T_max=200,
            eta_min=0.,
            last_epoch=-1,
            prefix="cosine learning policy: T_max == epochs - 1:"
        )
    )
}






