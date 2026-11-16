import torch
import torch.nn.functional as F
from torch import nn


def extract(values, timesteps, shape):
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, beta_end=0.1, diff_steps=100):
        super().__init__()
        self.denoise_fn = denoise_fn
        betas = torch.linspace(1e-4, beta_end, diff_steps)
        alphas = 1 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        previous = F.pad(cumulative[:-1], (1, 0), value=1)

        self.num_timesteps = diff_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - cumulative).sqrt())
        variance = betas * (1 - previous) / (1 - cumulative)
        self.register_buffer("posterior_log_variance_clipped", variance.clamp_min(1e-20).log())
        self.register_buffer("posterior_mean_coef1", betas * previous.sqrt() / (1 - cumulative))
        self.register_buffer(
            "posterior_mean_coef2", (1 - previous) * alphas.sqrt() / (1 - cumulative)
        )

    def q_sample(self, clean, timesteps, noise):
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )

    def log_prob(self, clean, cond, adjacency):
        timesteps = torch.randint(self.num_timesteps, (clean.shape[0],), device=clean.device)
        noisy = self.q_sample(clean, timesteps, torch.randn_like(clean))
        reconstructed = self.denoise_fn(noisy, timesteps, cond, adjacency)
        return F.mse_loss(reconstructed, clean)

    @torch.no_grad()
    def sample(self, sample_shape, cond, adjacency):
        value = torch.randn(sample_shape, device=self.betas.device)
        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (sample_shape[0],), step, device=value.device, dtype=torch.long
            )
            reconstructed = self.denoise_fn(value, timesteps, cond, adjacency).clamp_(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, timesteps, value.shape) * reconstructed
                + extract(self.posterior_mean_coef2, timesteps, value.shape) * value
            )
            if step:
                variance = extract(self.posterior_log_variance_clipped, timesteps, value.shape)
                value = mean + (0.5 * variance).exp() * torch.randn_like(value)
            else:
                value = mean
        return value
