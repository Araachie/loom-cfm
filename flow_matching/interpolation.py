from typing import Any
import torch

from lutils.dict_wrapper import DictWrapper


def sample_timestamps_uniform(x: torch.Tensor) -> torch.Tensor:
    return torch.rand(x.size(0), device=x.device)


def sample_timestamps_beta(x: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Beta(0.5, 0.5).sample(torch.Size((x.size(0),))).to(x.device)


def sample_timestamps_u_shaped(x: torch.Tensor) -> torch.Tensor:
    u = sample_timestamps_uniform(x)
    u = 0.5 - 0.25 * torch.asinh(torch.sinh(torch.tensor([2.0], device=x.device)) * (1 - 2 * u))
    return u


def sample_timestamps(x: torch.Tensor, t_dist: str) -> torch.Tensor:
    if t_dist == "uniform":
        return sample_timestamps_uniform(x)
    elif t_dist == "beta":
        return sample_timestamps_beta(x)
    elif t_dist == "u_shaped":
        return sample_timestamps_u_shaped(x)
    else:
        raise NotImplementedError


def sample_interpolation(
        source: torch.Tensor,
        target: torch.Tensor,
        timestamps: torch.Tensor,
        source_sigma: float = 0.000001,
        target_sigma: float = None) -> DictWrapper[str, Any]:
    # Set target sigma equal to the source sigma if none is specified
    if target_sigma is None:
        target_sigma = source_sigma

    utimestamps = timestamps.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Calculate interpolation
    input_points = source * (1 - utimestamps) + target * utimestamps
    sigma_inter = (1 - utimestamps) * source_sigma + utimestamps * target_sigma
    input_points += sigma_inter * torch.randn_like(input_points, device=input_points.device)

    # Calculate velocity
    target_vectors = target - source

    return DictWrapper(
        timestamps=timestamps,
        input_points=input_points,
        target_vectors=target_vectors,
    )
