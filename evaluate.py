import sys
sys.dont_write_bytecode = True

from argparse import ArgumentParser, Namespace
import os

import torch
import torchvision
import torchvision.transforms as T

from dataset import ImageNetDataset, IndexedDatasetWithFlips, H5Dataset
from evaluation.evaluator import Evaluator
from model import Model

from lutils.configuration import Configuration
from lutils.dict_wrapper import dict2namespace
from lutils.logger import Logger

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")

    return parser.parse_args()


def main(args: Namespace):
    device = torch.device("cuda:0")

    config = Configuration(args.config)

    extended_run_name = "{}_run-{}".format(config["name"], "evaluation")
    logger = Logger(
        project="straighter_flow_matching",
        run_name=extended_run_name,
        use_wandb=False,
        config=config,
        rank=0)

    # Load dataset
    logger.info("Loading data")
    path = os.path.join(config["data"]["data_root"], "val")
    tr = T.Compose([
        T.ToTensor(),
    ])
    dataset_name = config["data"]["name"]
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=tr, download=True)
    elif dataset_name == "imagenet":
        dataset = ImageNetDataset(root=path, train=False, transform=tr)
    elif dataset_name.startswith("imagefolder"):
        dataset = torchvision.datasets.ImageFolder(root=path, transform=tr)
    elif dataset_name.startswith("h5"):
        dataset = H5Dataset(root=path, transform=tr)
    dataset = IndexedDatasetWithFlips(
        dataset=dataset,
        flip_p=0.5 if config["data"]["random_flip"] else 0.0)

    # Setup model
    logger.info("Building the model and loading from ckpt")
    model = Model(config=dict2namespace(config))
    model.load_from_ckpt(args.ckpt)
    model.to(device)
    model.eval()

    # Setup evaluator
    logger.info("Instantiating evaluator object")
    evaluator = Evaluator(
        config=config["evaluation"],
        dataset=dataset,
        device=device)

    evaluator.evaluate(model, logger, log_media=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)