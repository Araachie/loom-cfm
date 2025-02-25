import sys
sys.dont_write_bytecode = True

from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import os

import torch
from model import Model
from model.generation import generate

from dataset.h5 import HDF5Maker

from lutils.configuration import Configuration
from lutils.dict_wrapper import dict2namespace
from lutils.logger import Logger
from lutils.logging import to_image


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory to write images to.")
    parser.add_argument("--num_images", type=int, required=True, help="Number of images to generate.")
    parser.add_argument("--method", type=str, required=True, help="Generation method of format solver-steps.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

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

    # Setup model
    logger.info("Building the model and loading from ckpt")
    model = Model(config=dict2namespace(config))
    model.load_from_ckpt(args.ckpt)
    model.to(device)
    model.eval()

    h5_maker = HDF5Maker(args.output, num_per_shard=100000, force=False)

    num_generated = 0
    bar = tqdm(total=args.num_images)
    os.makedirs(args.output, exist_ok=True)
    while num_generated < args.num_images:
        generated_images, _, source_noises = generate(
            model=model,
            batch_size=min(args.batch_size, args.num_images - num_generated),
            device=device,
            dtype=torch.float32,
            source=config["evaluation"]["source"],
            odesolver=args.method.split("-")[0],
            num_steps=int(args.method.split("-")[1]),
            first_step=None,
            return_source=True,
        )

        for source, target in zip(source_noises, generated_images):
            target_to_save = to_image(target).cpu().numpy()
            source_to_save = source.cpu().numpy()

            dict_to_save = {
                "source": source_to_save,
                "target": target_to_save,
            }

            h5_maker.add_data(dict_to_save, dtype="float32")
            num_generated += 1
            bar.update(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)