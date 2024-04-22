from typing import Literal

from ._samplers import _BlobSampler, _TriangleSampler


class Sampler:
    def __init__(
        self, geometry: Literal["blobs", "triangles"] = "blobs", seed=None, device="cpu"
    ):
        match geometry:
            case "blobs":
                self.sampler = iter(_BlobSampler(seed=seed, device=device))
            case "triangles":
                self.sampler = iter(_TriangleSampler(seed=seed, device=device))
            case _:
                raise ValueError(f"Invalid geometry type: {geometry}")

    def sample(self):
        return next(self.sampler)
