#!/usr/bin/env python


"""
Transfer Attack: utilize the source_model to attack 
the target model...
"""

import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME

METHOD = "Transfer"
FMT = "{description}={attack}-{epsilon:.4f}-{stepsize}-{steps}"

parser = argparse.ArgumentParser()
parser.add_argument("source_model", type=str)
parser.add_argument("source_path", type=str)
parser.add_argument("target_model", type=str)
parser.add_argument("target_path", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("--source_filename", type=str, default=SAVED_FILENAME)
parser.add_argument("--target_filename", type=str, default=SAVED_FILENAME)

# adversarial settings
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)

# basic settings
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='tensor,none')
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
    from src.base import FBDefense, FBAdversary
    from src.utils import load, set_seed, set_logger
    from src.utils import set_seed, activate_benchmark, load, set_logger
    from models.base import ADArch

    cfg = Config()
    
    # generate the log path
    mix_model = opts.source_model + "---" + opts.target_model
    _, cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset,
        model=mix_model, description=opts.description
    )
    # set logger
    logger = set_logger(
        path=cfg.log_path, 
        log2file=opts.log2file, 
        log2console=opts.log2console
    )
    logger.debug(f"source path: {opts.source_path}")
    logger.debug(f"target path: {opts.target_path}")

    activate_benchmark(opts.benchmark)
    set_seed(opts.seed)

    # load the source_model
    source_model = load_model(opts.source_model)(num_classes=get_num_classes(opts.dataset))
    mean, std = load_normalizer(opts.dataset)
    source_model = ADArch(model=source_model, mean=mean, std=std)
    load(
        model=source_model,
        path=opts.source_path,
        filename=opts.source_filename
    )

    # load the target_model
    target_model = load_model(opts.target_model)(num_classes=get_num_classes(opts.dataset))
    mean, std = load_normalizer(opts.dataset)
    target_model = ADArch(model=target_model, mean=mean, std=std)
    load(
        model=target_model, 
        path=opts.target_path,
        filename=opts.target_filename
    )

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset,
        transforms=opts.transform,
        train=False
    )
    cfg['testloader'] = load_dataloader(
        dataset=testset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )

    # set the attacker
    attack = load_fb_attack(
        attack_type=opts.attack,
        stepsize=opts.stepsize, 
        steps=opts.steps
    )

    cfg['attacker'] = FBAdversary(
        model=source_model, attacker=attack, 
        epsilon=opts.epsilon
    )

    # set the defender ...
    cfg['defender'] = FBDefense(
        model=target_model
    )

    return cfg


@timemeter("Main")
def main(defender, attacker, testloader, log_path):
    from src.criteria import TransferClassification
    from src.utils import distance_lp, getLogger
    logger = getLogger()
    running_success = 0.
    running_distance_linf = 0.
    running_distance_l2 = 0.
    for inputs, labels in testloader:
        inputs = inputs.to(attacker.device)
        labels = labels.to(attacker.device)

        criterion = TransferClassification(defender, labels)
        _, clipped, is_adv = attacker(inputs, criterion)
        inputs_ = inputs[is_adv]
        clipped_ = clipped[is_adv]

        dim_ = list(range(1, inputs.dim()))
        running_distance_linf += distance_lp(inputs_, clipped_, p=float("inf"), dim=dim_).sum().item()
        running_distance_l2 += distance_lp(inputs_, clipped_, p=2, dim=dim_).sum().item()
        running_success += is_adv.sum().item()

    running_distance_linf /= running_success
    running_distance_l2 /= running_success
    running_success /= len(testloader.dataset)

    results = "Success: {0:.3%}, Linf: {1:.5f}, L2: {2:.5f}".format(
        running_success, running_distance_linf, running_distance_l2
    )
    logger.info(results)


if __name__ == "__main__":
    from src.utils import readme
    cfg = load_cfg()
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)

































