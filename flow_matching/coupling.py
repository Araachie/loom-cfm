from typing import Tuple, Any, Union
import os
from itertools import chain

import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

from .distributions import SOURCE_DISTRIBUTIONS
from .interpolation import sample_interpolation, sample_timestamps
from lutils.dict_wrapper import DictWrapper


class BaseCoupling(object):
    def __init__(self, source: str):
        super(BaseCoupling, self).__init__()

        self.source_distribution = SOURCE_DISTRIBUTIONS[source]

    def sample(
            self,
            target: torch.Tensor,
            source: torch.Tensor = None,
            *args,
            **kwargs) -> DictWrapper[str, Any]:
        # Sample from source
        if source is None:
            source = self.source_distribution(target)

        # Sample timestamps
        kwargs["timestamps"] = sample_timestamps(source, kwargs.get("t_dist", "uniform"))

        # Couple source and target points
        coupling_output = self.couple(source, target, *args, **kwargs)

        # Interpolate
        interpolation_output = sample_interpolation(
            source=coupling_output["source"],
            target=coupling_output["target"],
            timestamps=coupling_output.get("timestamps", kwargs["timestamps"]),
            source_sigma=kwargs.get("source_sigma", 0.000001),
            target_sigma=kwargs.get("target_sigma", None),
        )

        # Build output dict
        interpolation_output["coupling_aux_output"] = coupling_output.get("aux_output", DictWrapper())

        return interpolation_output

    def couple(
            self,
            source: torch.Tensor,
            target: torch.Tensor,
            *args,
            **kwargs) -> DictWrapper[str, Any]:
        pass


class IndependentCoupling(BaseCoupling):

    def couple(
            self,
            source: torch.Tensor,
            target: torch.Tensor,
            *args,
            **kwargs) -> DictWrapper[str, Any]:
        return DictWrapper(
            source=source,
            target=target)


class OptimalTransportCoupling(BaseCoupling):
    def __init__(self, source: str):
        super(OptimalTransportCoupling, self).__init__(source)

    def couple(self, source: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> DictWrapper[str, Any]:
        rank = kwargs["rank"]
        world_size = kwargs["world_size"]
        # Solve optimal transport
        if world_size == 1:
            cost = torch.sqrt(((source.unsqueeze(1) - target.unsqueeze(0)) ** 2).sum([2, 3, 4]))
            _, assignment = linear_sum_assignment(cost.cpu().numpy())
            target = target[assignment]
        else:
            b = source.size(0)
            sources = self.gather(source, world_size)
            targets = self.gather(target, world_size)
            cost = torch.sqrt(((sources.unsqueeze(1) - targets.unsqueeze(0)) ** 2).sum([2, 3, 4]))
            _, assignment = linear_sum_assignment(cost.cpu().numpy())
            target = targets[assignment[b * rank:b * rank + b]]

        return DictWrapper(
            source=source,
            target=target)

    @staticmethod
    def gather(x: torch.Tensor, world_size: int) -> torch.Tensor:
        xs = [torch.zeros_like(x, device=x.device) for _ in range(world_size)]
        torch.distributed.all_gather(xs, x)
        xs = torch.cat(xs, dim=0)
        return xs


class SeedCachedOptimalTransportCoupling(BaseCoupling):
    def __init__(self, source: str, cache_dir: str, num_caches: int, no_swaps: bool = False):
        super(SeedCachedOptimalTransportCoupling, self).__init__(source)

        self.cache_dir = cache_dir
        self.num_caches = num_caches
        self.no_swaps = no_swaps
        os.makedirs(self.cache_dir, exist_ok=True)

    def couple(
            self,
            source: torch.Tensor,
            target: torch.Tensor,
            *args,
            **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], Any]:
        rank = kwargs["rank"]
        world_size = kwargs["world_size"]
        indices = kwargs["indices"]
        resample = kwargs["resample"]

        # Read from cache
        cached_seeds_paths = [os.path.join(self.cache_dir, f"{i.item():09d}.pt") for i in indices]
        cached_seeds_exist = torch.Tensor([os.path.exists(path) for path in cached_seeds_paths]).to(torch.bool)

        # Load sources
        cached_seeds = torch.stack([
            torch.load(path).cpu() if (exists and not resample)
            else torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, [self.num_caches])
            for path, exists in zip(cached_seeds_paths, cached_seeds_exist)])
        selected_indices = torch.randint(0, self.num_caches, [cached_seeds.size(0)])
        selected_seeds = cached_seeds[torch.arange(cached_seeds.size(0)), selected_indices]
        source = torch.stack([
            torch.zeros_like(source[0]).cpu().normal_(generator=torch.Generator().manual_seed(s.item())).to(source.device)
            for s in selected_seeds
        ])

        # PRE-COST
        pre_cost = self.calculate_cost(source, target)

        # Solve optimal transport
        if world_size == 1:
            cost = torch.sqrt(((source.unsqueeze(0) - target.unsqueeze(1)) ** 2).sum([-1, -2, -3]))
            if not self.no_swaps:
                _, assignment = linear_sum_assignment(cost.cpu().numpy())
            else:
                assignment = np.arange(source.size(0))
            source = source[assignment]
            new_seeds = selected_seeds[assignment]
        else:
            b = source.size(0)
            sources = self.gather(source, world_size)
            targets = self.gather(target, world_size)
            selected_seeds = self.gather(selected_seeds.to(source.device), world_size)
            cost = torch.sqrt(((sources.unsqueeze(0) - targets.unsqueeze(1)) ** 2).sum([-1, -2, -3]))
            if not self.no_swaps:
                _, assignment = linear_sum_assignment(cost.cpu().numpy())
            else:
                assignment = np.arange(sources.size(0))
            source = sources[assignment[b * rank:b * rank + b]]
            new_seeds = selected_seeds[assignment[b * rank:b * rank + b]].cpu()

        # Calculate number of swaps for logging
        num_swaps = (assignment != np.arange(assignment.shape[0])).astype(np.float32).sum()

        # POST-COST
        post_cost = self.calculate_cost(source, target)

        # Write to cache
        cached_seeds[torch.arange(cached_seeds.size(0)), selected_indices] = new_seeds
        for i, path in enumerate(cached_seeds_paths):
            torch.save(cached_seeds[i].cpu(), path)

        # Auxiliary output
        aux_output = DictWrapper(
            num_swaps=num_swaps,
            pre_cost=pre_cost,
            post_cost=post_cost)

        return DictWrapper(
            source=source,
            target=target,
            aux_output=aux_output)

    @staticmethod
    def calculate_cost(x: torch.Tensor, y: torch.Tensor):
        return torch.sqrt(((x - y) ** 2).sum([-1, -2, -3])).mean()

    @staticmethod
    def gather(x: torch.Tensor, world_size: int) -> torch.Tensor:
        xs = [torch.zeros_like(x, device=x.device) for _ in range(world_size)]
        torch.distributed.all_gather(xs, x)
        xs = torch.cat(xs, dim=0)
        return xs


class MixedCoupling(BaseCoupling):
    def __init__(self, source: str, first: str, first_params: dict, second: str, second_params: dict, p: float = 0.5):
        super(MixedCoupling, self).__init__(source)

        self.first = COUPLINGS[first](source) if first_params is None else COUPLINGS[first](source, **first_params)
        self.second = COUPLINGS[second](source) if second_params is None else COUPLINGS[second](source, **second_params)
        self.p = p

    def couple(self, source: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> DictWrapper[str, Any]:
        if torch.rand(size=[]) < self.p:
            return self.first.couple(source, target, *args, **kwargs)
        else:
            return self.second.couple(source, target, *args, **kwargs)


class TimestampMixedCoupling(BaseCoupling):
    def __init__(self, source: str, power: float, first: str, first_params: dict, second: str, second_params: dict):
        super(TimestampMixedCoupling, self).__init__(source)

        self.power = power
        self.first = COUPLINGS[first](source) if first_params is None else COUPLINGS[first](source, **first_params)
        self.second = COUPLINGS[second](source) if second_params is None else COUPLINGS[second](source, **second_params)

    def couple(self, source: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> DictWrapper[str, Any]:
        t = kwargs["timestamps"]
        p = self.power * t ** (self.power - 1) - (self.power - 1) * t ** self.power
        c = torch.rand(source.size(0), device=source.device)

        first_output_dict = self.first.couple(source, target, *args, **kwargs)
        second_output_dict = self.second.couple(source, target, *args, **kwargs)

        mask = (c < p).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        source = torch.where(mask, first_output_dict["source"], second_output_dict["source"])
        target = torch.where(mask, first_output_dict["target"], second_output_dict["target"])

        output_dict = DictWrapper(
            source=source,
            target=target,
            timestamps=t,
        )

        for k, v in chain(first_output_dict.items(), second_output_dict.items()):
            if k not in ["source", "target"]:
                output_dict[k] = v

        return output_dict


COUPLINGS = {
    "independent": IndependentCoupling,
    "ot": OptimalTransportCoupling,
    "ot_cached_seed": SeedCachedOptimalTransportCoupling,
    "mixed": MixedCoupling,
    "t_mixed": TimestampMixedCoupling,
}
