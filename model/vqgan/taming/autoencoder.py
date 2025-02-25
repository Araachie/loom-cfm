import importlib

import torch
import torch.nn as nn

import numpy as np

from model.vqgan.taming.modules import Encoder, Decoder
from model.vqgan.taming.quantize import VectorQuantizer2 as VectorQuantizer


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# https://ommer-lab.com/files/latent-diffusion/vq-f8.zip
vq_f8_ddconfig = dict(embed_dim=4, n_embed=16384, double_z=False, z_channels=4, resolution=256, in_channels=3,
                      out_ch=3, ch=128, ch_mult=[1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[32], dropout=0.0)


vq_f8_small_ddconfig = dict(embed_dim=4, n_embed=16384, double_z=False, z_channels=4, resolution=64, in_channels=3,
                      out_ch=3, ch=128, ch_mult=[1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)


# https://heibox.uni-heidelberg.de/f/0e42b04e2e904890a9b6/?dl=1
vq_f16_ddconfig = dict(embed_dim=8, n_embed=16384, double_z=False, z_channels=8, resolution=256, in_channels=3,
                       out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)


kl_f8_ddconfig = dict(embed_dim=4, double_z=True, z_channels=4, resolution=256, in_channels=3,
                      out_ch=3, ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[32], dropout=0.0)


class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(ddconfig["n_embed"], ddconfig["embed_dim"], beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], ddconfig["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["embed_dim"], ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


class VQModelInterface(VQModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*ddconfig["embed_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(ddconfig["embed_dim"], ddconfig["z_channels"], 1)
        self.embed_dim = ddconfig["embed_dim"]

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior


class KLModelInterface(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.first_batch = True

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()

        if self.first_batch:
            self.first_batch = False
            world_size = kwargs.get("world_size", 1.0)
            z_mean = z.mean() / world_size
            torch.distributed.all_reduce(z_mean, op=torch.distributed.ReduceOp.SUM)
            z_var = torch.pow(z - z_mean, exponent=2).mean() / world_size
            torch.distributed.all_reduce(z_var, op=torch.distributed.ReduceOp.SUM)
            self.scale.data.fill_(torch.sqrt(z_var).item())

        return z / self.scale

    def decode(self, z):
        z = z * self.scale
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
