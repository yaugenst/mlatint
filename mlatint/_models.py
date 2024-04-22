import pytorch_lightning as pl
import torch
import torch.distributions
import torch.nn as nn


class VAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim,
        channels_encode,
        channels_decode,
        input_dim,
        output_dim,
        kld_weight,
        kld_weight_annealing,
        bin_weight,
        bin_cutoff,
        bin_weight_annealing,
        lr,
        weight_decay,
        steps,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = self.get_encoder(input_dim, channels_encode, latent_dim)
        self.decoder = self.get_decoder(latent_dim, channels_decode, output_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        p, q, z = self._sample(x)
        return self.decoder(z)

    @staticmethod
    def _gaussian_likelihood(x_hat, x, logscale):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(x_hat, scale)
        log_pxz = dist.log_prob(x)
        return torch.flatten(log_pxz, 1).mean(1)

    @staticmethod
    def _kl_divergence(p, q, z):
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kld = log_qzx - log_pz
        return kld.mean(-1)

    @staticmethod
    def _binarization(x, cutoff):
        b = -torch.log10(torch.sum(4 * x * (1 - x)) / float(x.numel()))
        return torch.minimum(b, cutoff) / cutoff

    @staticmethod
    def _sigmoid_anneal(x, minimum, maximum, slope, offset):
        return minimum + (maximum - minimum) / (
            1 + torch.exp(torch.tensor(-slope * (x - offset)))
        )

    def _get_annealed_weight(self, batch_idx, weight, params):
        if params is None or weight == 0.0:
            return weight
        else:
            minimum, slope, offset = params
            offset *= self.hparams.steps
            return self._sigmoid_anneal(batch_idx, minimum, weight, slope, offset)

    def _sample(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        log_var = torch.clamp(log_var, -10, 10)
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x = batch.unsqueeze(1)
        x_enc = self.encoder(x)
        p, q, z = self._sample(x_enc)
        x_hat = self.decoder(z)

        # Note: Reconstruction loss is scaled by input size, KL divergence is
        # scaled by latent size. We get rid of this scaling by taking the mean
        # instead of the sum, which would correspond to the mathematical definition.
        # This makes the weighting of the terms independent of latent and input sizes.
        reconstruction_loss = -self._gaussian_likelihood(x_hat, x, self.log_scale)

        kld = self._kl_divergence(p, q, z)
        kld_weight = self._get_annealed_weight(
            batch_idx, self.hparams.kld_weight, self.hparams.kld_weight_annealing
        )

        cutoff = torch.tensor(self.hparams.bin_cutoff).type_as(x)
        binarization = self._binarization(x_hat, cutoff)
        bin_weight = self._get_annealed_weight(
            batch_idx, self.hparams.bin_weight, self.hparams.bin_weight_annealing
        )

        elbo = kld_weight * kld + reconstruction_loss - bin_weight * binarization
        elbo = elbo.mean()  # batch mean

        logs = {
            "reconstruction_loss": reconstruction_loss.mean(),
            "kld": kld.mean(),
            "binarization": binarization,
            "kld_weight": kld_weight,
            "bin_weight": bin_weight,
            "hp_metric": elbo,
            "elbo": elbo,
        }
        return elbo, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(logs, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=10 * self.hparams.lr,
            total_steps=self.hparams.steps,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class VAE2d(VAE):
    @staticmethod
    def get_encoder(input_dim, channels, output_dim):
        return Encoder2d(input_dim, channels, output_dim)

    @staticmethod
    def get_decoder(input_dim, channels, output_dim):
        return Decoder2d(input_dim, channels, output_dim)


class Encoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()

        self.act = nn.SELU()

        self.conv_in = self.get_conv(
            1, channels[0], kernel_size=1, stride=1, padding=0, bias=True
        )

        blocks = []
        for ii, channel in enumerate(channels[:-1]):
            blocks += [
                self.get_conv(
                    channel,
                    channels[ii + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                ),
                self.get_bn(channels[ii + 1]),
                self.act,
            ]
        self.conv_blocks = nn.Sequential(*blocks)

        self.linear = nn.Sequential(
            nn.Linear(
                channels[-1] * int(input_dim / 2 ** (len(channels) - 1)) ** self.dim,
                output_dim,
            ),
            self.act,
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_blocks(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class Encoder2d(Encoder):
    dim = 2

    @staticmethod
    def get_conv(in_channels, out_channels, kernel_size, stride, padding, bias):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm2d(features)


class Decoder(nn.Module):
    def __init__(self, input_dim, channels, output_dim):
        super().__init__()

        self.act = nn.SELU()

        self.linear = nn.Sequential(
            nn.Linear(
                input_dim,
                channels[0] * int(output_dim / 2 ** (len(channels) - 1)) ** self.dim,
            ),
            self.act,
        )

        self.unflatten = self.get_unflatten(channels, output_dim)

        blocks = []
        for ii, channel in enumerate(channels[:-1]):
            blocks += [
                self.get_conv_transpose(
                    channel,
                    channels[ii + 1],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                    bias=False,
                ),
                self.get_bn(channels[ii + 1]),
                self.act,
            ]
        self.conv_blocks = nn.Sequential(*blocks)

        self.conv_out = self.get_conv(channels[-1], 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.conv_blocks(x)
        x = self.conv_out(x)
        x = torch.sigmoid(x)
        return x


class Decoder2d(Decoder):
    dim = 2

    @staticmethod
    def get_conv_transpose(
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias
    ):
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    @staticmethod
    def get_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    @staticmethod
    def get_bn(features):
        return nn.BatchNorm2d(features)

    @staticmethod
    def get_unflatten(channels, output_dim):
        return nn.Unflatten(
            dim=1,
            unflattened_size=(
                channels[0],
                int(output_dim / 2 ** (len(channels) - 1)),
                int(output_dim / 2 ** (len(channels) - 1)),
            ),
        )
