import cairo
import numpy as np
import pytorch_lightning as pl

from scipy.ndimage import gaussian_filter
import torch
from numpy.random import SeedSequence
from torch.utils.data import IterableDataset


class _Sampler(IterableDataset):
    def __init__(self, *args, **kwargs):
        super(self).__init__(*args, **kwargs)

    def __iter__(self):
        num_workers, worker_id = self._get_worker_info()
        ss = SeedSequence(self.seed)
        rng = [np.random.default_rng(s) for s in ss.spawn(num_workers)][worker_id]
        return iter(self._make_iter(rng))

    def _get_worker_info(self):
        if (info := torch.utils.data.get_worker_info()) is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = info.num_workers
            worker_id = info.id
        return num_workers, worker_id

    def _make_iter(self, rng):
        while True:
            yield self._make(rng)


class _BlobSampler(_Sampler):
    def __init__(self, shape=(128, 128), sigma=12, seed=None, device="cpu"):
        self.shape = shape
        self.sigma = sigma
        self.seed = seed
        self.device = device

    def _make(self, rng):
        while True:
            design = rng.random(self.shape, dtype="f4")
            design = gaussian_filter(design, self.sigma, mode="constant") > 0.5
            if not np.all(~design):
                break
        return torch.as_tensor(design, dtype=torch.float, device=self.device)


class _TriangleSampler(_Sampler):
    def __init__(
        self,
        shape=(128, 128),
        smooth=True,
        coverage=0.6,
        total_width=5,
        seed=None,
        device="cpu",
    ):
        self.shape = shape
        self.seed = seed
        self.device = device
        self.smooth = smooth
        self.total_width = total_width
        self.drawable_width = coverage * self.total_width

    def _make(self, rng):
        with cairo.ImageSurface(cairo.Format.RGB24, *self.shape) as surface:
            context = cairo.Context(surface)
            context.scale(*self.shape)
            context.translate(0.5, 0.5)
            context.scale(1 / self.total_width, 1 / self.total_width)
            if not self.smooth:
                context.set_antialias(cairo.Antialias.NONE)
            else:
                context.set_antialias(cairo.Antialias.BEST)
            context.set_line_width(1.25 * self.total_width / self.shape[0])
            self._draw_triangles(context, rng)
            data = np.array(surface.get_data(), dtype="u8").reshape(*self.shape, 4)
            shape = torch.as_tensor(
                data[:, :, 0] / 255.0, dtype=torch.float, device=self.device
            )
        return shape

    def _draw_triangles(self, ctx, rng):
        def draw_triangle(x, y, angle, size):
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(angle)
            ctx.move_to(-size / 2, -size / 2)
            ctx.rel_line_to(size, 0)
            ctx.rel_line_to(-size / 2, size)
            ctx.rel_line_to(-size / 2, -size)
            ctx.stroke_preserve()
            ctx.fill()
            ctx.restore()

        ctx.set_source_rgb(1.0, 1.0, 1.0)

        ctx.save()
        ctx.rotate(rng.uniform(0, 2 * np.pi))
        x = y = angle = 0
        size = 0.75
        for i in range(15):
            if (
                x < -self.drawable_width / 2
                or x > self.drawable_width / 2
                or y < -self.drawable_width / 2
                or y > self.drawable_width / 2
            ):
                break
            draw_triangle(x, y, angle, size)
            c = rng.integers(4)
            if c == 0:
                x += size / 2
                angle = np.pi if angle == 0 else 0
            elif c == 1:
                x -= size / 2
                angle = np.pi if angle == 0 else 0
            elif c == 2:
                y += size
                angle = np.pi if angle == 0 else 0
            elif c == 3:
                y -= size
                angle = np.pi if angle == 0 else 0
        ctx.restore()
