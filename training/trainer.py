import copy
import os
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from flow_matching import COUPLINGS

from lutils.configuration import Configuration
from lutils.constants import MAIN_PROCESS
from lutils.dict_wrapper import DictWrapper
from lutils.logger import Logger
from lutils.logging import to_video, make_observations_grid
from training.utils import check_ddp_consistency
from evaluation.evaluator import Evaluator
from dataset.utils import MultiEpochsDataLoader


class Trainer:
    """
    Class that handles the training
    """

    def __init__(
            self,
            rank: int,
            run_name: str,
            config: Configuration,
            dataset: Dataset,
            sampler: torch.utils.data.distributed.DistributedSampler,
            num_gpus: int,
            device: torch.device):
        """
        Initializes the Trainer

        :param rank: rank of the current process
        :param config: training configuration
        :param dataset: dataset to train on
        :param sampler: sampler to create the dataloader with
        :param device: device to use for training
        """
        super(Trainer, self).__init__()

        self.config = config
        self.rank = rank
        self.is_main_process = self.rank == MAIN_PROCESS
        self.num_gpus = num_gpus
        self.device = device

        # Create folder for saving
        self.run_path = os.path.join("./runs", run_name)
        os.makedirs(self.run_path, exist_ok=True)
        os.makedirs(os.path.join(self.run_path, "checkpoints"), exist_ok=True)

        # Setup dataloader
        self.dataset = dataset
        self.sampler = sampler
        self.dataloader = MultiEpochsDataLoader(
            dataset=dataset,
            batch_size=self.config["batching"]["batch_size"],
            shuffle=sampler is None,
            num_workers=self.config["batching"]["num_workers"],
            sampler=sampler,
            pin_memory=True)

        # Optimizer will be defined in train_epoch
        self.optimizer = None
        self.lr_scheduler = None

        self.global_step = 0

        # Setup cache params
        def check_and_set_cache_dir(q, coupling_key, params_key):
            if q[coupling_key].endswith("cached"):
                if q[params_key] is None:
                    q[params_key] = {}
                if "cache_dir" not in q[params_key]:
                    q[params_key]["cache_dir"] = os.path.join(self.run_path, "cache")
                self.cache_dir = q[params_key]["cache_dir"]

        self.cache_dir = None
        check_and_set_cache_dir(self.config, "coupling", "coupling_params")
        if self.config["coupling"].endswith("mixed"):
            check_and_set_cache_dir(self.config["coupling_params"], "first", "first_params")
            check_and_set_cache_dir(self.config["coupling_params"], "second", "second_params")

    def init_optimizer(self, model: nn.Module):
        if self.config["optimizer"]["optimizer"] == "Adam":
            optimizer_class = torch.optim.Adam
        else:
            raise NotImplementedError
        self.optimizer = optimizer_class(
            model.parameters(),
            lr=self.config["optimizer"]["learning_rate"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(0.9, 0.999))
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.config["optimizer"]["num_warmup_steps"],
            num_training_steps=self.config["num_epochs"] * len(self.dataloader),
            power=self.config["optimizer"]["scheduler_p"])

    def get_lr(self):
        assert self.optimizer is not None

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(
            self,
            model: nn.Module,
            logger: Logger,
            evaluator: Evaluator,
            scalar_logging_frequency: int = 100,
            evaluation_frequency: int = 50000,
            saving_frequency: int = 5000,
            checkpointing_frequency: int = 50000):
        """
        Trains the model for one epoch

        """

        model.train()
        dmodel = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

        # Setup optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)
        scaler = torch.cuda.amp.GradScaler(enabled=self.config["use_fp16"])

        # Initialize the EMA model
        ema_model = copy.deepcopy(dmodel)

        # Setup coupling
        if self.config["coupling_params"] is not None:
            coupling = COUPLINGS[self.config["coupling"]](self.config["source"], **(self.config["coupling_params"]))
        else:
            coupling = COUPLINGS[self.config["coupling"]](self.config["source"])

        # Setup loading bar
        for epoch in tqdm(range(self.config["num_epochs"]), desc=f"Epochs", disable=not self.is_main_process):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            train_gen = tqdm(self.dataloader, desc=f"Batches", disable=not self.is_main_process, leave=False)

            resample_sources = epoch % self.config["resample_every"] == 0
            train_epoch = epoch % self.config["resample_every"] >= self.config["train_after"]

            for indices, batch in train_gen:
                # Fetch data
                if isinstance(batch, (tuple, list)):
                    source = batch[0].cuda()
                    target = batch[1].cuda()
                else:
                    source = None
                    target = batch.cuda()

                # Encode target
                target = dmodel.encode(target, world_size=self.num_gpus)

                # Sample data for training step
                coupling_output = coupling.sample(
                    target=target,
                    source=source,
                    indices=indices,
                    rank=self.rank,
                    world_size=self.num_gpus,
                    resample=resample_sources,
                    source_sigma=self.config["sigma"],
                    target_sigma=self.config.get("target_sigma", None),
                    t_dist=self.config.get("t_dist", "uniform"),
                )
                coupling_aux_output = coupling_output["coupling_aux_output"]
                timestamps = coupling_output["timestamps"]
                input_points = coupling_output["input_points"]
                target_vectors = coupling_output["target_vectors"]

                if train_epoch:
                    # Zero gradients
                    model.zero_grad()

                    with torch.amp.autocast(device_type="cuda",
                                            dtype=torch.float16 if self.config["use_fp16"] else torch.float32):
                        # Forward the model
                        predicted_vectors = model(input_points, timestamps)

                        model_outputs = DictWrapper(
                            target_vectors=target_vectors,
                            predicted_vectors=predicted_vectors)

                        # Compute the loss
                        loss, auxiliary_output = self.calculate_loss(model_outputs)

                    # Backward pass
                    scaler.scale(loss).backward()

                    # Uncomment this if nans appear in the loss during the training
                    # self.reduce_gradients(model, self.num_gpus)

                    # Optimizer step
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["optimizer"]["grad_clip"])
                    scaler.step(self.optimizer)
                    self.lr_scheduler.step()

                    # Update scaler for the next iteration
                    scaler.update()

                    # Update the EMA model
                    self.update_ema(
                        source=dmodel,
                        target=ema_model,
                        decay=self.config["ema_decay"],
                        restore_keys=["ae.scale"])

                    # Log scalars
                    if self.global_step % scalar_logging_frequency == 0 and self.is_main_process:
                        self.log_scalars(
                            auxiliary_output if train_epoch else DictWrapper(),
                            coupling_aux_output,
                            model,
                            logger)

                    # Evaluate
                    if self.global_step != 0 and self.global_step % evaluation_frequency == 0 and self.is_main_process:
                        evaluator.evaluate(ema_model, logger)

                    # Finalize logs
                    logger.finalize_logs(step=self.global_step)

                    # Save checkpoint
                    if self.global_step % checkpointing_frequency == 0:
                        self.save_checkpoint(ema_model, f"step_{self.global_step}.pth")
                    elif self.global_step % saving_frequency == 0:
                        self.save_checkpoint(ema_model)

                    self.global_step += 1

            # Close loading bar
            train_gen.close()

        # Final evaluation
        logger.info("Final evaluation")
        if self.is_main_process:
            model.eval()
            evaluator.evaluate(ema_model, logger)

            logger.finalize_logs(step=self.global_step)

        # Save the model
        logger.info("Saving the trained model...")
        self.save_checkpoint(model, f"final_step_{self.global_step}.pth")

    def calculate_loss(
            self,
            results: DictWrapper[str, Any]) -> Tuple[torch.Tensor, DictWrapper[str, Any]]:
        """
        Calculates the loss

        :param results: Dict with the model outputs
        :return: [1,] The loss value
        """

        # Flow matching loss
        flow_matching_loss = \
            torch.pow(results.predicted_vectors - results.target_vectors, exponent=2).mean([1, 2, 3])

        # Sum up all the losses
        loss_weights = self.config["loss_weights"]
        loss = \
            loss_weights["flow_matching_loss"] * flow_matching_loss

        # DDP hack
        def add_zero_to_loss(value):
            if v is None:
                return loss
            return loss + value.mul(0).mean()

        for _, v in results.items():
            if isinstance(v, list):
                for ev in v:
                    loss = add_zero_to_loss(ev)
            else:
                loss = add_zero_to_loss(v)

        # Create auxiliary output
        auxiliary_output = DictWrapper(
            # Total loss
            total_loss=loss.mean(),
            loss_var=torch.pow(loss, 2).mean(),

            # Loss terms
            flow_matching_loss=flow_matching_loss.mean(),
        )

        return loss.mean(), auxiliary_output

    def log_scalars(
            self,
            loss_terms: DictWrapper[str, Any],
            other_data: DictWrapper[str, Any],
            model: nn.Module,
            logger: Logger):
        for k, v in loss_terms.items():
            logger.log(f"Training/Loss/{k}", v)

        # Log training stats
        logger.log(f"Training/Stats/learning_rate", self.get_lr())
        # logger.log(f"Training/Stats/total_loss_is_nan", torch.isnan(loss_terms.total_loss).to(torch.int8))
        # logger.log(f"Training/Stats/total_loss_is_inf", torch.isinf(loss_terms.total_loss).to(torch.int8))

        # Other stats
        for k, v in other_data.items():
            logger.log(f"Training/Stats/{k}", v)

    @staticmethod
    def reduce_gradients(model: nn.Module, num_gpus: int):
        params = [param for param in model.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    @staticmethod
    def update_ema(source, target, decay, restore_keys):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            if key in restore_keys:
                target_dict[key].data.copy_(
                    source_dict[key].data
                )
            else:
                target_dict[key].data.copy_(
                    target_dict[key].data * decay + source_dict[key].data * (1 - decay)
                )

    def mean_reduce(self, value):
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value / self.num_gpus

    def save_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if self.num_gpus > 1:
            check_ddp_consistency(model, r".*\..+_(mean|var|tracked)")

        if self.is_main_process:
            state_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "model": model.state_dict(),
                "global_step": self.global_step
            }
            if checkpoint_name:
                torch.save(state_dict, os.path.join(self.run_path, "checkpoints", checkpoint_name))
            torch.save(state_dict, os.path.join(self.run_path, "checkpoints", "latest.pth"))

    def load_checkpoint(self, model: nn.Module, checkpoint_name: str = None):
        if checkpoint_name is None:
            checkpoint_name = "latest.pth"
        filename = os.path.join(self.run_path, "checkpoints", checkpoint_name)
        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        # Init optimizer and scheduler if not yet
        if self.optimizer is None:
            self.init_optimizer(model)

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        loaded_state = torch.load(filename, map_location=map_location)
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])

        is_state_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_state_ddp = True
                break

        if is_state_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        else:
            state = loaded_state["model"]

        dmodel = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        dmodel.load_state_dict(state)

        self.global_step = loaded_state["global_step"]
