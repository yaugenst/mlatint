from importlib.resources import files
from pathlib import Path
from typing import Literal

import torch
from einops import rearrange

from ._models import VAE2d

_data_dir = files("mlatint.pretrained_models")


class VAE:
    def __init__(
        self,
        model: Literal["blobs", "triangles"] = "blobs",
        device: str = "cpu",
    ):
        model_fp = _data_dir.joinpath(f"vae2d_{model}.ckpt")
        self.model = VAE2d.load_from_checkpoint(model_fp, map_location=device)
        self.model.freeze()
        self.model.eval()

    def encode(self, x):
        return self.model.encoder(x[None, None]).squeeze()

    def decode(self, x):
        z = self.model._sample(x[None])[-1]
        return self.model.decoder(z).squeeze()

    def __call__(self, x):
        return self.model(x[None]).squeeze()


class FNO:
    def __init__(
        self, device: str = "cpu", eps_min: float = 1.0, eps_max: float = 2.25
    ):
        model_fp = _data_dir.joinpath("fno2d_16k_maxwell.pt")
        self.model = torch.jit.load(model_fp, map_location=device)

        self.eps_min = eps_min
        self.eps_max = eps_max

    def __call__(self, x):
        x = self.eps_min + x * (self.eps_max - self.eps_min)
        x = self.model(x[None, None])
        x = rearrange(x, "b ReIm x y -> b x y ReIm")
        x = torch.view_as_complex(x.contiguous())
        return x.squeeze()
