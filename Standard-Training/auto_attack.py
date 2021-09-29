#!/usr/bin/env python

from typing import Tuple
import torch
import argparse
from src.loadopts import *
from autoattack import AutoAttack



METHOD = "AutoAttack"
FMT = "{description}={norm}-{version}-{epsilon}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)

# for AA
parser.add_argument("--norm", choices=("Linf", "L2"), default="Linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--version", choices=("standard", "plus"), default="standard")
parser.add_argument("-b", "--batch_size", type=int, default=256)

parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.utils import gpu, load, set_seed, set_logger

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

    set_seed(opts.seed)

    # load the model
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    model.set_normalizer(load_normalizer(opts.dataset))
    device = gpu(model)
    load(
        model=model, 
        path=opts.info_path,
        device=device
    )
    model.eval()

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset, 
        transform='None',
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
        device=device
    )

    return cfg, log_path


def main(attacker, data, targets):
    attacker.run_standard_evaluation(data, targets, bs=opts.batch_size)



if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import readme
    cfg, log_path = load_cfg()
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()


