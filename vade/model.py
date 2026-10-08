"""Diagonal Gaussian mixture VAE and the negative ELBO (paper Eq. 12)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class VaDE(nn.Module):
    def __init__(self, input_dim=784, n_clusters=10, latent_dim=10,
                 hidden_dims=(500, 500, 2000), reconstruction="bce", alpha=1.0):
        super().__init__()
        self.config = dict(input_dim=input_dim, n_clusters=n_clusters,
                           latent_dim=latent_dim, hidden_dims=list(hidden_dims),
                           reconstruction=reconstruction, alpha=alpha)
        self.reconstruction = reconstruction
        self.alpha = alpha
        self.encoder = self._mlp([input_dim, *hidden_dims], final_relu=True)
        self.z_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.z_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = self._mlp([latent_dim, *reversed(hidden_dims), input_dim])
        self.mixture_logits = nn.Parameter(torch.zeros(n_clusters))
        self.cluster_means = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        self.cluster_log_vars = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _mlp(dims, final_relu=False):
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2 or final_relu:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h).clamp(-20, 20)

    def decode(self, z):
        output = self.decoder(z)
        return output.sigmoid() if self.reconstruction == "bce" else output

    def log_responsibilities(self, z, legacy_prior=False):
        log_var = self.cluster_log_vars.clamp(-20, 20)
        log_density = -0.5 * (math.log(2 * math.pi) + log_var[None]
                              + (z[:, None] - self.cluster_means[None]).square()
                              * (-log_var[None]).exp()).sum(-1)
        # Historical implementation counted log(pi) once per latent dimension.
        factor = z.shape[-1] if legacy_prior else 1
        return F.log_softmax(log_density + factor * F.log_softmax(self.mixture_logits, 0), -1)

    def reconstruction_loss(self, output, x):
        if self.reconstruction == "bce":
            return F.binary_cross_entropy_with_logits(output, x, reduction="none").sum(-1)
        return (output - x).square().sum(-1)

    def loss(self, x):
        mean, log_var = self.encode(x)
        z = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        reconstruction = self.reconstruction_loss(self.decoder(z), x)
        log_gamma = self.log_responsibilities(z)
        gamma = log_gamma.exp()
        prior_log_var = self.cluster_log_vars.clamp(-20, 20)
        # Gaussian KL to each component; constants cancel with q(z|x) entropy.
        gaussian_kl = 0.5 * (prior_log_var[None] - log_var[:, None]
                            + (log_var.exp()[:, None]
                               + (mean[:, None] - self.cluster_means[None]).square())
                            * (-prior_log_var[None]).exp() - 1).sum(-1)
        categorical_kl = (gamma * (log_gamma - F.log_softmax(self.mixture_logits, 0))).sum(-1)
        kl = (gamma * gaussian_kl).sum(-1) + categorical_kl
        loss = (self.alpha * reconstruction + kl).mean()
        return loss, {"reconstruction": reconstruction.mean().detach(), "kl": kl.mean().detach()}

    def mixture_parameters(self):
        return [self.mixture_logits, self.cluster_means, self.cluster_log_vars]
