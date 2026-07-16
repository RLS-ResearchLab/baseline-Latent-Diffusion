import torch


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def q_sample(self, z0, t, noise):
        # z0: (B, N, D), t: (B,), noise: (B, N, D) -> noisy latent at step t
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ac * z0 + sqrt_1mac * noise

    @torch.no_grad()
    def ddim_sample(self, model, shape, y, steps=50, eta=0.0, device="cpu"):
        # shape: (B, N, D), y: (B,) class ids -> denoised latent z0
        step_indices = torch.linspace(self.timesteps - 1, 0, steps, device=device).long()
        z = torch.randn(shape, device=device)

        for i, t in enumerate(step_indices):
            t_batch = t.expand(shape[0])
            eps = model(z, t_batch, y, train=False)

            ac_t = self.alphas_cumprod[t]
            ac_prev = self.alphas_cumprod[step_indices[i + 1]] if i + 1 < len(step_indices) else torch.tensor(1.0, device=device)

            z0_pred = (z - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            dir_z = (1 - ac_prev).sqrt() * eps
            z = ac_prev.sqrt() * z0_pred + dir_z

        return z
