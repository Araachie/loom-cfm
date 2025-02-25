import sys
import time

import torchvision.utils

sys.dont_write_bytecode = True

from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from PIL import Image
import os

import torch
from model import Model
from model.generation import generate

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
    parser.add_argument("--nrows", type=int, default=8, help="nrows in make_grid.")
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
    if model.ae is not None:
        model.ae.cpu()
        del model.ae.encoder
    model.eval()

    # Generate images
    logger.info("Generating images")
    num_generated = 0
    bar = tqdm(total=args.num_images)
    os.makedirs(args.output, exist_ok=True)
    samples = []
    while num_generated < args.num_images:
        num_samples = min(args.batch_size, args.num_images - num_generated)
        generated_images, _ = generate(
            model=model,
            batch_size=num_samples,
            device=device,
            dtype=torch.float32,
            source=config["evaluation"]["source"],
            odesolver=args.method.split("-")[0],
            num_steps=int(args.method.split("-")[1]),
            first_step=None,
            return_source=False,
            decode=True,
        )

        samples.append(to_image(generated_images).cpu())
        bar.update(num_samples)
        num_generated += num_samples
    bar.close()
    samples = torch.cat(samples, dim=0)

    # Saving images
    samples_grid = torchvision.utils.make_grid(samples, nrow=args.nrows)
    samples_image = Image.fromarray(samples_grid.permute(1, 2, 0).numpy())
    output_samples_path = os.path.join(args.output, f"samples_{args.method}.png")
    samples_image.save(output_samples_path)
    logger.info(f"Samples saved at {output_samples_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)