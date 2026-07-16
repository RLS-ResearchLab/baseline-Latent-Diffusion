import torch
import torch.nn.functional as F


def vae_loss(recon, target, mu, logvar, kl_weight=1e-6):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl, recon_loss, kl


def diffusion_loss(model, diffusion, z0, y, train=True):
    # z0: (B, N, D) clean latents, y: (B,) class ids
    B = z0.shape[0]
    t = torch.randint(0, diffusion.timesteps, (B,), device=z0.device)
    noise = torch.randn_like(z0)
    z_t = diffusion.q_sample(z0, t, noise)
    pred = model(z_t, t, y, train=train)
    # training the model to predict the added noise so far
    return F.mse_loss(pred, noise) 
