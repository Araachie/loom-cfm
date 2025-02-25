import torch
import torch.nn as nn

from einops import rearrange

from torchdiffeq import odeint_adjoint as odeint


class Sampler(object):
    def __init__(self, model: nn.Module, dtype):
        super(Sampler, self).__init__()

        self.model = model
        self.dtype = dtype
        self.num_steps = 0

    def __call__(self, t, x):
        timestamps = torch.full([x.size(0)], t, dtype=x.dtype, device=x.device)
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            predicted_vectors = self.model(x, timestamps)
        return predicted_vectors.to(x.dtype)

    def callback_step(self, t0, y0, dt):
        self.num_steps += 1


def sample(y, t0, t1, model, method="euler", dtype=torch.float32,
           num_steps=10, first_step=0.1, return_trajectory=False):
    f = Sampler(model, dtype)

    t = torch.cat([
        torch.tensor([t0], device=y.device),
        torch.linspace(t0 + first_step, t1, steps=num_steps, device=y.device)], dim=0)
    y = odeint(
        f, y, t, method=method, atol=1e-5, rtol=1e-5, adjoint_params=model.parameters())

    if return_trajectory:
        return y, f.num_steps
    else:
        return y[-1], f.num_steps


@torch.no_grad()
def generate(
        model: nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        source: str = "normal",
        odesolver: str = "euler",
        num_steps: int = 20,
        first_step: float = None,
        return_source: bool = False,
        return_trajectory: bool = False,
        decode: bool = True):
    """

    :param model: predicts the vector field given x and t
    :param batch_size: how many samples to generate
    :param device: the device to perform the computations on
    :param dtype: data type to use
    :param source: source distribution
    :param odesolver: the solver to use for the ode integration
    :param num_steps: number of steps for the fixed grid solvers
    :param first_step: the size of the first step
    :param return_source: whether to return the source noise
    :param return_trajectory: whether to return the whole sampling trajectory
    :param decode: whether to decode the latents
    :return: [b c h w] generated samples
    """

    c = model.backbone.in_channels
    h = w = model.backbone.image_size

    # Move the model to device
    model.to(device)

    # Sample initial noise TODO: replace with function from flow_matching.distributions
    if source == "normal":
        y0 = torch.randn([batch_size, c, h, w]).to(device)
    elif isinstance(source, torch.Tensor):
        y0 = source
    else:
        raise NotImplementedError

    # Generate
    if first_step is None:
        first_step = 1.0 / num_steps
    generated, num_steps = sample(
        y0, 0.0, 1.0, model, method=odesolver, dtype=dtype,
        num_steps=num_steps, first_step=first_step, return_trajectory=return_trajectory)

    # Decode generated
    if decode:
        if return_trajectory:
            generated = rearrange(generated, "n b c h w -> (n b) c h w")
        generated = model.decode(generated)
        if return_trajectory:
            generated = rearrange(generated, "(n b) c h w -> n b c h w", b=batch_size)

    if return_source:
        return generated, num_steps, y0
    return generated, num_steps
