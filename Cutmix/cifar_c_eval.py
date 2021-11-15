#!/usr/bin/env python


from typing import Iterable, List
import torch
import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME, DEVICE


METHOD = "Corruption"
FMT = "{description}={corruption_type}"

CORRUPTIONS = [
    "brightness", "defocus_blur", "fog", "gaussian_blur", "glass_blur", "jpeg_compression",
    "motion_blur", "saturate", "snow", "speckle_noise", "contrast", "elastic_transform", "frost",
    "gaussian_noise", "impulse_noise", "pixelate", "shot_noise", "spatter", "zoom_blur"
]

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)
parser.add_argument("--filename", type=str, default=SAVED_FILENAME)

# corruption
parser.add_argument("-ct","--corruption_type", choices=CORRUPTIONS + ['all'], default="all")

# basic settings
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("--progress", action="store_false", default=True, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--benchmark", action="store_false", default=True, 
                help="cudnn.benchmark == True ?")
parser.add_argument("-m", "--description", type=str, default=METHOD)
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



@timemeter("Setup")
def load_cfg() -> 'Config':
    from src.dict2obj import Config
    from src.utils import set_seed, activate_benchmark, load, set_logger
    from models.base import ADArch

    cfg = Config()
   
    # generate the log path
    _, cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset+'c',
        model=opts.model, description=opts.description
    )
    # set logger
    logger = set_logger(
        path=cfg.log_path, 
        log2file=opts.log2file, 
        log2console=opts.log2console
    )
    logger.debug(opts.info_path)

    activate_benchmark(opts.benchmark)
    set_seed(opts.seed)

    # load the model
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    mean, std = load_normalizer(opts.dataset)
    model = ADArch(model=model, mean=mean, std=std)
    load(
        model=model, 
        path=opts.info_path,
        filename=opts.filename
    )
    cfg['model'] = model

    return cfg

@timemeter("Dataset/Corruption")
def load_cifar_c(corruption: str) -> Iterable:
    import torchvision.transforms as T
    from src.datasets import CIFAR10C, CIFAR100C
    from src.utils import getLogger

    getLogger().info(f"==================Corruption Type: {corruption}==================")
    if opts.dataset == "cifar10":
        dataset = CIFAR10C(
            corruption_type=corruption,
            transform=T.ToTensor()
        )
    elif opts.dataset == "cifar100":
        dataset = CIFAR100C(
            corruption_type=corruption,
            transform=T.ToTensor()
        )
    else:
        raise NotImplementedError(f"Supported: CIFAR-10|100-C: cifar10|cifar100")
    testloader = load_dataloader(
        dataset=dataset,
        batch_size=opts.batch_size,
        train=False,
        show_progress=opts.progress
    )
    return testloader


@timemeter("Main")
def main(model, log_path, device=DEVICE):

    from src.utils import AverageMeter, getLogger

    corruptions: List
    if opts.corruption_type == 'all':
        corruptions = CORRUPTIONS
    else:
        corruptions = [opts.corruption_type]
    
    acc_meter = AverageMeter('Accuracy', fmt=".3%")
    running_accuracy = []
    for corruption in corruptions:
        acc_meter.reset()
        testloader = load_cifar_c(corruption)
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            with torch.no_grad():
                logits = model(inputs)
            preds = logits.argmax(-1) == labels

            acc_meter.update(preds.sum().item(), n=inputs.size(0), mode="sum")
            getLogger().info(acc_meter)
        running_accuracy.append(acc_meter.avg)


    running_accuracy = ', '.join([f"{acc:.3%}" for acc in running_accuracy])
    getLogger().info(f"Accuracy: {running_accuracy}")
   

if __name__ == "__main__":
    from src.utils import readme
    cfg = load_cfg()
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)








