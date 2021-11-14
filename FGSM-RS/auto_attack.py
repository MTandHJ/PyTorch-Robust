#!/usr/bin/env python

from typing import Tuple
import torch
import argparse
from src.loadopts import *
from src.config import SAVED_FILENAME, DEVICE
from src.utils import timemeter
from autoattack import AutoAttack



METHOD = "AutoAttack"
FMT = "{description}={norm}-{version}-{epsilon}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)
parser.add_argument("--filename", type=str, default=SAVED_FILENAME)

# for AA
parser.add_argument("--norm", choices=("Linf", "L2", "L1"), default="Linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--version", choices=("standard", "plus"), default="standard")

# basic settings
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("--transform", type=str, default='tensor,none')
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
def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.utils import set_seed, activate_benchmark, load, set_logger
    from models.base import ADArch

    cfg = Config()

    # generate the log path
    _, log_path = generate_path(METHOD, opts.dataset, 
                        opts.model, opts.description)
    # set logger
    logger = set_logger(
        path=log_path,
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
    model.eval()

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset, 
        transforms=opts.transform,
        train=False
    )
    data = []
    targets = []
    for i in range(len(testset)):
        img, label = testset[i]
        data.append(img)
        targets.append(label)
    
    cfg['data'] = torch.stack(data)
    cfg['targets'] = torch.tensor(targets, dtype=torch.long)


    cfg['attacker'] = AutoAttack(
        model,
        norm=opts.norm,
        eps=opts.epsilon,
        version=opts.version,
        device=DEVICE,
    )

    return cfg, log_path


@timemeter('Main')
def main(attacker, data, targets):
    attacker.run_standard_evaluation(data, targets, bs=opts.batch_size)



if __name__ == "__main__":
    from src.utils import readme
    cfg, log_path = load_cfg()
    readme(log_path, opts, mode="a")

    main(**cfg)



