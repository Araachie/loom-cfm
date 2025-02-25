import os
from argparse import Namespace as ArgsNamespace
from typing import Any, Dict

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as T

from lutils.configuration import Configuration
from lutils.dict_wrapper import dict2namespace
from lutils.logger import Logger
from dataset import IndexedDatasetWithFlips, ImageNetDataset, PairedDataset, H5Dataset
from model import Model
from training.trainer import Trainer
from evaluation.evaluator import Evaluator as EvaluatorFID


def setup_training_arguments(args: ArgsNamespace) -> Dict[str, Any]:
    training_args = dict()

    # Load config file
    training_args["config"] = Configuration(args.config)

    # Other args
    training_args["run_name"] = args.run_name
    training_args["random_seed"] = args.random_seed
    training_args["num_gpus"] = args.num_gpus
    training_args["resume_step"] = args.resume_step
    training_args["use_wandb"] = args.wandb

    return training_args


def training_loop(
        rank: int,
        config: Configuration,
        run_name: str,
        resume_step: int = None,
        cudnn_benchmark: bool = True,
        random_seed: int = None,
        num_gpus: int = 1,
        use_wandb: bool = False):
    # Initialize some stuff
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Initialize logger
    extended_run_name = "{}_run-{}".format(config["name"], run_name)
    logger = Logger(
        project="loom-cfm",
        run_name=extended_run_name,
        use_wandb=use_wandb,
        config=config,
        rank=rank)
    logger.silent = not use_wandb

    # Load dataset
    logger.info("Loading data")
    datasets = {}

    for key in ["train", "val"]:
        path = os.path.join(config["data"]["data_root"], key)
        if key == "train":
            tr = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            tr = T.Compose([
                T.ToTensor(),
            ])

        dataset_name = config["data"]["name"]
        if dataset_name.startswith("reflow") and key == "train":
            dataset = PairedDataset(root=path, transform=tr)
        if dataset_name.startswith("reflow") and key == "val":
            dataset_name = dataset_name[7:]
            path = os.path.join(config["data"]["eval_data_root"], key)
        if dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root=path, train=key == "train", transform=tr, download=True)
        elif dataset_name.startswith("imagenet"):
            dataset = ImageNetDataset(root=path, train=key == "train", transform=tr)
        elif dataset_name.startswith("imagefolder"):
            dataset = torchvision.datasets.ImageFolder(root=path, transform=tr)
        elif dataset_name.startswith("h5"):
            dataset = H5Dataset(root=path, transform=tr)
        else:
            raise ValueError("Unknown dataset")
        datasets[key] = IndexedDatasetWithFlips(
            dataset=dataset,
            flip_p=0.5 if config["data"]["random_flip"] else 0.0)

    # Setup model and distribute across gpus
    logger.info("Building the model and distributing it across gpus")
    model = Model(config=dict2namespace(config))
    model.to(device)
    if "init_from_checkpoint" in config["model"]:
        model.load_from_ckpt(config["model"]["init_from_checkpoint"])
    if (num_gpus > 1) and len(list(model.parameters())) != 0:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=False)

    # Setup trainer
    logger.info("Instantiating trainer object")
    sampler = torch.utils.data.distributed.DistributedSampler(
        datasets["train"],
        drop_last=True) if num_gpus > 1 else None
    trainer = Trainer(
        rank=rank,
        run_name=extended_run_name,
        config=config["training"],
        dataset=datasets["train"],
        sampler=sampler,
        num_gpus=num_gpus,
        device=device)

    # Setup evaluator
    logger.info("Instantiating evaluator object")
    evaluator = EvaluatorFID(
        config=config["evaluation"],
        dataset=datasets["val"],
        device=device)

    # Resume training if needed
    if resume_step == -1:
        logger.info("Loading the latest checkpoint")
        trainer.load_checkpoint(model)
    elif resume_step is not None:
        logger.info(f"Loading the checkpoint at step {resume_step}")
        trainer.load_checkpoint(model, f"step_{resume_step}.pth")

    # Launch the training loop
    logger.info("Launching training loop")
    trainer.train(model=model, logger=logger, evaluator=evaluator)
