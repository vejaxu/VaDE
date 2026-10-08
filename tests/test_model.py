import numpy as np
import torch
from torch.distributions import Categorical, Independent, Normal, kl_divergence

from vade.metrics import evaluate
from vade.model import VaDE


def test_responsibilities_match_distribution_reference():
    torch.manual_seed(7)
    model = VaDE(input_dim=4, n_clusters=3, latent_dim=2, hidden_dims=(8,))
    with torch.no_grad():
        model.mixture_logits.copy_(torch.tensor([-2., 1., 0.]))
        model.cluster_means.normal_()
        model.cluster_log_vars.normal_()
    z = torch.randn(11, 2)
    component = Independent(Normal(model.cluster_means, (model.cluster_log_vars / 2).exp()), 1)
    reference = (component.log_prob(z[:, None]) + model.mixture_logits.log_softmax(0)).softmax(-1)
    torch.testing.assert_close(model.log_responsibilities(z).exp(), reference)


def test_elbo_matches_analytic_kl_and_updates_mixture():
    torch.manual_seed(5)
    model = VaDE(input_dim=4, n_clusters=3, latent_dim=2, hidden_dims=(8,))
    with torch.no_grad():
        model.cluster_means.normal_()
        model.cluster_log_vars.normal_()
    x = torch.rand(7, 4)  # A non-full batch is valid.
    torch.manual_seed(9)
    mean, log_var = model.encode(x)
    z = mean + torch.randn_like(mean) * (log_var / 2).exp()
    gamma = model.log_responsibilities(z).exp()
    q = Independent(Normal(mean[:, None], (log_var[:, None] / 2).exp()), 1)
    p = Independent(Normal(model.cluster_means, (model.cluster_log_vars / 2).exp()), 1)
    expected = (model.reconstruction_loss(model.decoder(z), x)
                + (gamma * kl_divergence(q, p)).sum(-1)
                + kl_divergence(Categorical(probs=gamma), Categorical(logits=model.mixture_logits))).mean()
    torch.manual_seed(9)
    loss, _ = model.loss(x)
    torch.testing.assert_close(loss, expected)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.mixture_parameters()]
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, model.mixture_parameters()))


def test_clustering_metrics_ignore_label_permutation():
    metrics, _, _ = evaluate(np.array([10, 10, 20, 20, 30]), np.array([2, 2, 0, 0, 1]))
    assert metrics == dict(f1_macro=1., nmi=1., ari=1.)
