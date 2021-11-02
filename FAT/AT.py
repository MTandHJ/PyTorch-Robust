#!/usr/bin/env python


from typing import Tuple
import argparse
from src.loadopts import *
from src.utils import timemeter



METHOD = "AT"
SAVE_FREQ = 5
FMT = "{description}={learning_policy}-{optimizer}-{lr}-{weight_decay}" \
        "={epsilon:.4f}-{stepsize}-{steps}" \
        "={tau}-{omega}-{random_type}-{dynamictau}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)

# adversarial training settings
parser.add_argument("--epsilon", type=float, default=0.031)
parser.add_argument("--stepsize", type=float, default=0.007, 
                help="abs_stepsize")
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--tau", type=int, default=0)
parser.add_argument("--omega", type=int, default=0.001,
                help="random sample parameter for adv data generation")
parser.add_argument("--random-type", type=str, default='uniform')
parser.add_argument("--dynamictau", action="store_false", default=True,
                help="whether to use dynamic tau")

# basic settings
parser.add_argument("--loss", type=str, default="cross_entropy")
parser.add_argument("--optimizer", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=2e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
parser.add_argument("-lp", "--learning_policy", type=str, default="FAT-AT", 
                help="the learning policy defined in config.py")
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentations which will be applied during training.")

# the ratio of valid dataset
parser.add_argument("--ratio", type=float, default=.0,
                help="the ratio of validation; use testset if ratio is 0.")

# eval
parser.add_argument("--eval-train", action="store_true", default=False)
parser.add_argument("--eval-valid", action="store_false", default=True)
parser.add_argument("--eval-freq", type=int, default=1,
                help="for valid dataset only")

parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("-m", "--description", type=str, default=METHOD)
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)




@timemeter("Setup")
def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import Coach, AdversaryForTrain
    from src.utils import gpu, set_seed, load_checkpoint, set_logger
    from src.attacks import LinfFriendlyPGD

    cfg = Config()

    # generate the path for logging information and saving parameters
    cfg['info_path'], cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.model, description=opts.description
    )
    # set logger
    logger = set_logger(
        path=cfg.log_path, 
        log2file=opts.log2file, 
        log2console=opts.log2console
    )

    set_seed(opts.seed)

    # the model and other settings for training
    model = load_model(opts.model)(num_classes=get_num_classes(opts.dataset))
    model.set_normalizer(load_normalizer(opts.dataset))
    device = gpu(model)

    # load the dataset
    trainset, validset = load_dataset(
        dataset_type=opts.dataset,
        transforms=opts.transform,
        ratio=opts.ratio,
        train=True
    )
    if opts.ratio == 0:
        logger.warning(
            "[Warning] The ratio of validation is 0. Use testset instead."
        )
        validset = load_dataset(
            dataset_type=opts.dataset,
            transforms="tensor,none",
            train=False
        )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset,
        batch_size=opts.batch_size,
        train=True,
        show_progress=opts.progress
    )
    cfg['validloader'] = load_dataloader(
        dataset=validset,
        batch_size=opts.batch_size,
        train=False,
        show_progress=opts.progress
    )

    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optimizer, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, 
        learning_policy_type=opts.learning_policy,
        T_max=opts.epochs
    )

    
    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    cfg['coach'] = Coach(
        model=model, device=device, 
        loss_func=load_loss_func(opts.loss), 
        optimizer=optimizer, 
        learning_policy=learning_policy
    )

    # set the attack
    cfg['attack'] = LinfFriendlyPGD(
        model=model,
        rel_stepsize="none",
        abs_stepsize=opts.stepsize,
        steps=opts.steps, epsilon=opts.epsilon,
        tau=opts.tau, omega=opts.emega,
        random_start=True, random_type=opts.random_type,
        dynamictau=opts.dynamictau,
        bounds=(0, 1)
    )

    cfg['valider'] = load_valider(
        model=model, device=device, dataset_type=opts.dataset
    )

    return cfg


def preparation(valider):
    from src.utils import TrackMeter, ImageMeter, getLogger
    from src.dict2obj import Config
    logger = getLogger()
    acc_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )
    rob_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )

    acc_logger.plotter = ImageMeter(*acc_logger.values(), title="Accuracy")
    rob_logger.plotter = ImageMeter(*rob_logger.values(), title="Robustness")

    @timemeter("Evaluation")
    def evaluate(dataloader, prefix='Valid', epoch=8888):
        acc_nat, acc_adv = valider.evaluate(dataloader)
        logger.info(f"{prefix} >>> [TA: {acc_nat:.3%}]    [RA: {acc_adv:.3%}]")
        getattr(acc_logger, prefix.lower())(data=acc_nat, T=epoch)
        getattr(rob_logger, prefix.lower())(data=acc_adv, T=epoch)
        return acc_nat, acc_adv

    return acc_logger, rob_logger, evaluate


@timemeter("Main")
def main(
    coach, attacker, valider, 
    trainloader, validloader, start_epoch, 
    info_path, log_path
):  

    from src.utils import save_checkpoint

    # preparation
    acc_logger, rob_logger, evaluate = preparation(valider)

    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        if epoch % opts.eval_freq == 0:
            if opts.eval_train:
                evaluate(trainloader, prefix='Train', epoch=epoch)
            if opts.eval_valid:
                acc_nat, acc_rob = evaluate(validloader, prefix="Valid", epoch=epoch)
                coach.check_best(acc_nat, acc_rob, info_path, epoch=epoch)

        running_loss = coach.adv_train(trainloader, attacker, epoch=epoch)

    # save the model
    coach.save(info_path)

    # final evaluation
    evaluate(trainloader, prefix='Train', epoch=opts.epochs)
    acc_nat, acc_rob = evaluate(validloader, prefix="Valid", epoch=opts.epochs)
    coach.check_best(acc_nat, acc_rob, info_path, epoch=opts.epochs) 

    acc_logger.plotter.plot()
    rob_logger.plotter.plot()
    acc_logger.plotter.save(log_path)
    rob_logger.plotter.save(log_path)




if __name__ ==  "__main__":
    from src.utils import readme
    cfg = load_cfg()
    opts.log_path = cfg.log_path
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")

    main(**cfg)



