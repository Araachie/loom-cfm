import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torcheval.metrics import FrechetInceptionDistance as FID

from lutils.configuration import Configuration
from lutils.logger import Logger
from model.generation import generate


class Evaluator:
    """
    Class that handles the evaluation of the FID metrics
    """

    def __init__(
            self,
            config: Configuration,
            dataset: Dataset,
            device: torch.device):
        """
        Initializes the Trainer

        :param config: training configuration
        :param dataset: dataset to train on
        :param device: device to use for training
        """
        super(Evaluator, self).__init__()

        self.config = config
        self.batch_size = self.config["batching"]["batch_size"]
        self.device = device

        # Setup dataloader
        self.dataset = dataset
        sampler = torch.utils.data.RandomSampler(
            self.dataset,
            replacement=True,
            num_samples=self.config["num_real_samples_for_fid"],
            generator=torch.Generator().manual_seed(42))
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config["batching"]["num_workers"],
            sampler=sampler,
            pin_memory=True)

        # Define metrics
        self.reals_obtained = False
        self.fid = FID(device=device)

    @torch.no_grad()
    def evaluate(
            self,
            model: nn.Module,
            logger: Logger,
            log_media: bool = True):
        """

        :param model: model to evaluate
        :param logger: performs logging
        :param log_media: whether to log the generated images (wandb logging has to be enabled)
        :return:
        """

        dmodel = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        metrics = {}
        samples = {}
        for steps in self.config["steps_to_evaluate"]:
            if isinstance(steps, list):
                first_step = steps[1]
                steps = steps[0]
            else:
                first_step = None
            if not self.reals_obtained:
                # Update with real images
                real_gen = tqdm(self.dataloader, desc=f"Real Batches, Steps: {steps}", leave=False)
                for _, batch in real_gen:
                    self.fid.update(batch, is_real=True)
                self.reals_obtained = True

            # Update with fake images
            num_fake_batches = int(self.config["num_samples_for_fid"] / self.batch_size)
            fake_gen = tqdm(range(num_fake_batches), desc=f"Fake Batches, Steps: {steps}", leave=False)
            generated = None
            num_steps = 0
            for _ in fake_gen:
                generated, cur_num_steps = generate(
                    dmodel,
                    self.batch_size,
                    device=self.device,
                    dtype=torch.float16 if self.config["use_fp16"] else torch.float32,
                    source=self.config["source"],
                    odesolver="euler" if not isinstance(steps, str) else steps.split("-")[0],
                    num_steps=steps if not isinstance(steps, str) else int(steps.split("-")[1]),
                    first_step=first_step)
                generated = 0.5 * torch.clamp(generated, -1, 1) + 0.5
                self.fid.update(generated, is_real=False)
                num_steps += cur_num_steps

            steps_str = steps if first_step is None else f"{steps}_{first_step}"
            metrics[f"fid_{steps_str}"] = self.fid.compute()
            metrics[f"num_steps_{steps_str}"] = num_steps / num_fake_batches
            samples[f"generated_{steps_str}"] = generated

            self.reset_fakes()

        for k, v in metrics.items():
            logger.log(f"Evaluation/Metrics/{k}", v)
        if log_media:
            for k, v in samples.items():
                logger.log(f"Evaluation/Media/{k}", logger.wandb().Image(v))

    def reset_fakes(self):
        for state_name, default in self.fid._state_name_to_default.items():
            if "fake" in state_name:
                setattr(self.fid, state_name, default.clone().to(self.fid.device))
