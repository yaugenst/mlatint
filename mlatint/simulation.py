from functools import partial
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.constants import c

from ._simulation import fdfd2d_torch


class FDFD:
    def __init__(
        self,
        wavelength: float = 1.55,
        n1: float = 1.0,
        n2: float = 1.5,
        resolution: int = 25,
        npml: int = 12,
        l0: float = 1e-6,
    ):
        self.wavelength = wavelength
        self.n1 = n1
        self.n2 = n2
        self.resolution = resolution
        self.npml = npml
        self.l0 = l0

    @property
    def omega(self) -> float:
        return 2 * np.pi * c / (self.wavelength * self.l0)

    @property
    def dl(self) -> float:
        return 1 / self.resolution

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, 4 * (self.npml,))
        x = self.n1**2 + (self.n2**2 - self.n1**2) * x
        src = torch.zeros_like(x)
        src[self.npml - 1, :] = 1
        ez = fdfd2d_torch.apply(
            x, src, self.omega, self.dl, (self.npml, self.npml), self.l0
        )
        ez = F.pad(ez, 4 * (-self.npml,))
        return -ez  # somehow had the wrong sign in training
