import torch
import torch.nn as nn

from .unet import UNetModel
from .vqgan.taming.autoencoder import vq_f8_ddconfig, vq_f16_ddconfig, kl_f8_ddconfig, VQModelInterface, KLModelInterface

from lutils.configuration import Configuration


class Model(nn.Module):
    def __init__(self, config: Configuration):
        super(Model, self).__init__()

        self.ae = None
        if hasattr(config.model, "ae_checkpoint"):
            if config.model.ae_config == "vq_f8":
                ae_config = vq_f8_ddconfig
                ae_class = VQModelInterface
            elif config.model.ae_config == "vq_f16":
                ae_config = vq_f16_ddconfig
                ae_class = VQModelInterface
            elif config.model.ae_config == "kl_f8":
                ae_config = kl_f8_ddconfig
                ae_class = KLModelInterface
            else:
                raise ValueError("Only [f8, f16] are supported ae configs")

            self.ae = ae_class(ae_config, config.model.ae_checkpoint)
            self.ae.requires_grad_(False)
            self.ae.eval()

        self.backbone = UNetModel(
            image_size=config.model.input_size,
            in_channels=config.model.in_channels,
            model_channels=config.model.ch,
            out_channels=config.model.out_ch,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attn_resolutions,
            dropout=config.model.dropout,
            channel_mult=config.model.ch_mult,
            num_heads=config.model.num_heads,
            num_head_channels=config.model.num_head_channels,
            use_fp16=False,
            conv_resample=config.model.resamp_with_conv,
        )

    def load_from_ckpt(self, ckpt_path: str):
        loaded_state = torch.load(ckpt_path, map_location="cpu")

        is_state_ddp = False
        for k in loaded_state["model"]:
            if k.startswith("module"):
                is_state_ddp = True
                break

        is_self_ddp = isinstance(self, torch.nn.parallel.DistributedDataParallel)
        if is_state_ddp and not is_self_ddp:
            state = {k.replace("module.", ""): v for k, v in loaded_state["model"].items()}
        elif not is_state_ddp and is_self_ddp:
            state = {f"module.{k}": v for k, v in loaded_state["model"].items()}
        else:
            state = loaded_state["model"]

        dmodel = self.module if is_self_ddp else self
        dmodel.load_state_dict(state)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        :param x: [b C H W]
        :return: [b c h w]
        """

        return x if self.ae is None else self.ae.encode(x, **kwargs)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """

        :param z: [b c h w]
        :return: [b C H W]
        """

        return z if self.ae is None else self.ae.decode(z)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """

        :param x: [b c h w]
        :param t: [b]
        :return: [b c h w]
        """

        return self.backbone(x, t)
