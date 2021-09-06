#!/usr/bin/env python


from typing import Tuple
import argparse
from src.loadopts import *
from src.utils import timemeter


METHOD = "FGSM-AT"
SAVE_FREQ = 5
PRINT_FREQ = 20
FMT = "{description}={leverage}={learning_policy}-{optimizer}-{lr}" \
        "={attack}-{epsilon:.4f}-{stepsize}-{steps}" \
        "={batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)

# coefficient of penalty
parser.add_argument("--leverage", type=float, default=.5)

# adversarial training settings
parser.add_argument("--attack", type=str, default="fgsm")
parser.add_argument("--epsilon", type=float, default=8/255)
# no use settings for FGSM-AT
parser.add_argument("--stepsize", type=float, default=0.25, 
                help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=10)

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
parser.add_argument("-lp", "--learning_policy", type=str, default="AT", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied during training.")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--log2file", action="store_false", default=True,
                help="False: remove file handler")
parser.add_argument("--log2console", action="store_false", default=True,
                help="False: remove console handler if log2file is True ...")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="Train")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



@timemeter("Setup")
def load_cfg() -> Tuple[Config, str]:
    from src.dict2obj import Config
    from src.base import Coach, AdversaryForTrain
    from src.utils import gpu, set_seed, load_checkpoint, set_logger

    cfg = Config()

    # generate the path for logging information and saving parameters
    cfg['info_path'], cfg['log_path'] = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.model, description=opts.description
    )
    # set logger
    set_logger(
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
    trainset = load_dataset(
        dataset_type=opts.dataset, 
        transform=opts.transform, 
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset, 
        batch_size=opts.batch_size, 
        train=True,
        show_progress=opts.progress
    )
    validset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=False
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
    attack = load_attack(
        attack_type=opts.attack,
        stepsize=opts.stepsize, 
        steps=opts.steps
    )

    cfg['attacker'] = AdversaryForTrain(
        model=model, attacker=attack, 
        device=device, epsilon=opts.epsilon
    )

    cfg['valider'] = load_valider(
        model=model, device=device, dataset_type=opts.dataset
    )

    return cfg


@timemeter("Evaluation")
def evaluate(
    valider, trainloader, validloader,
    acc_logger, rob_logger, 
    logger, writter, log_path,
    epoch = 8888
):
    train_acc_nat, train_acc_adv = valider.evaluate(trainloader)
    valid_acc_nat, valid_acc_adv = valider.evaluate(validloader)

    logger.info(f"Train >>> [TA: {train_acc_nat:.3%}]    [RA: {train_acc_adv:.3%}]")
    logger.info(f"Test. >>> [TA: {valid_acc_nat:.3%}]    [RA: {valid_acc_adv:.3%}]")
    writter.add_scalars("Accuracy", {"train":train_acc_nat, "valid":valid_acc_nat}, epoch)
    writter.add_scalars("Robustness", {"train":train_acc_adv, "valid":valid_acc_adv}, epoch)

    acc_logger.train(data=train_acc_nat, T=epoch)
    acc_logger.valid(data=valid_acc_nat, T=epoch)
    rob_logger.train(data=train_acc_adv, T=epoch)
    rob_logger.valid(data=valid_acc_adv, T=epoch)



@timemeter("Main")
def main(
    coach, attacker, valider, 
    trainloader, validloader, start_epoch, 
    info_path, log_path
):  
    from src.utils import save_checkpoint, TrackMeter, ImageMeter, getLogger
    from src.dict2obj import Config
    logger = getLogger()
    acc_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )
    acc_logger.plotter = ImageMeter(*acc_logger.values(), title="Accuracy")

    rob_logger = Config(
        train=TrackMeter("Train"),
        valid=TrackMeter("Valid")
    )
    rob_logger.plotter = ImageMeter(*rob_logger.values(), title="Robustness")


    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        if epoch % PRINT_FREQ == 0:
            evaluate(
                valider=valider,
                trainloader=trainloader, validloader=validloader,
                acc_logger=acc_logger, rob_logger=rob_logger, 
                logger=logger, writter=writter,
                log_path=log_path, epoch=epoch
            )
            

        running_loss = coach.train(trainloader, attacker, leverage=opts.leverage, epoch=epoch)
        writter.add_scalar("Loss", running_loss, epoch)


    evaluate(
        valider=valider,
        trainloader=trainloader, validloader=validloader,
        acc_logger=acc_logger, rob_logger=rob_logger, 
        logger=logger, writter=writter,
        log_path=log_path, epoch=opts.epochs
    )

    acc_logger.plotter.plot()
    rob_logger.plotter.plot()
    acc_logger.plotter.save(writter)
    rob_logger.plotter.save(writter)


if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import readme
    cfg = load_cfg()
    readme(cfg.info_path, opts)
    readme(cfg.log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=cfg.log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()


