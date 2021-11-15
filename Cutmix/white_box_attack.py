#!/usr/bin/env python


import torch
import argparse
from src.loadopts import *
from src.utils import timemeter
from src.config import SAVED_FILENAME


METHOD = "WhiteBox"
FMT = "{description}={attack}-{epsilon_min:.4f}-{epsilon_max}-{epsilon_times}-{stepsize}-{steps}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)
parser.add_argument("--filename", type=str, default=SAVED_FILENAME)

# adversarial settings
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon_min", type=float, default=8/255)
parser.add_argument("--epsilon_max", type=float, default=1.)
parser.add_argument("--epsilon_times", type=int, default=1)
parser.add_argument("--stepsize", type=float, default=0.25, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)

# basic settings
parser.add_argument("-b", "--batch_size", type=int, default=256)
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
    from src.base import  FBAdversary
    from src.utils import set_seed, activate_benchmark, load, set_logger
    from models.base import ADArch

    cfg = Config()
   
    # generate the log path
    _, cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset,
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

    # the model and other settings for training
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    mean, std = load_normalizer(opts.dataset)
    model = ADArch(model=model, mean=mean, std=std)
    load(
        model=model, 
        path=opts.info_path,
        filename=opts.filename
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

    epsilons = torch.linspace(opts.epsilon_min, opts.epsilon_max, opts.epsilon_times).tolist()
    cfg['attacker'] = FBAdversary(
        model=model, attacker=attack, 
        epsilon=epsilons
    )

    return cfg


@timemeter("Main")
def main(attacker, testloader, log_path):
    from src.utils import distance_lp, getLogger
    logger = getLogger()
    running_success = [0.] * opts.epsilon_times
    running_distance_linf = [0.] * opts.epsilon_times
    running_distance_l2 = [0.] * opts.epsilon_times
    for inputs, labels in testloader:
        inputs = inputs.to(attacker.device)
        labels = labels.to(attacker.device)

        _, clipped, is_adv = attacker(inputs, labels)
        dim_ = list(range(1, inputs.dim()))
        for epsilon in range(opts.epsilon_times):
            inputs_ = inputs[is_adv[epsilon]]
            clipped_ = clipped[epsilon][is_adv[epsilon]]

            running_success[epsilon] += is_adv[epsilon].sum().item()
            running_distance_linf[epsilon] += distance_lp(inputs_, clipped_, p=float('inf'), dim=dim_).sum().item()
            running_distance_l2[epsilon] += distance_lp(inputs_, clipped_, p=2, dim=dim_).sum().item()

    datasize = len(testloader.dataset)
    for epsilon in range(opts.epsilon_times):
        running_distance_linf[epsilon] /= running_success[epsilon]
        running_distance_l2[epsilon] /= running_success[epsilon]
        running_success[epsilon] /= datasize

    running_accuracy = list(map(lambda x: 1. - x, running_success))

    running_accuracy = ', '.join([f"{acc:.3%}" for acc in running_accuracy])
    running_distance_linf = ', '.join([f"{dis_linf:.5f}" for dis_linf in running_distance_linf])
    running_distance_l2 = ', '.join([f"{dis_l2:.5f}" for dis_l2 in running_distance_l2])
   
    logger.info(f"Accuracy: {running_accuracy}")
    logger.info(f"Distance-Linf: {running_distance_linf}")
    logger.info(f"Distance-L2: {running_distance_l2}")
   

if __name__ == "__main__":
    from src.utils import readme
    cfg = load_cfg()
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)







