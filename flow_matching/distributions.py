import torch


def normal_(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.randn_like(x, device=x.device, **kwargs)


SOURCE_DISTRIBUTIONS = {
    "normal": normal_,
}
