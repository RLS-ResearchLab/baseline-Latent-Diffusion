import os
import sys
import yaml
import torch
from torchvision.utils import save_image

from src.models import build_model
from src.diffusion import GaussianDiffusion


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # build the VAE model, load the trained weights
    vae_ckpt = torch.load(config["train"]["vae_ckpt_path"], map_location="cuda")
    vae = build_model("vae", vae_ckpt["config"]).cuda()
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()

    # build the DiT model, load the trained weights 
    dit_ckpt = torch.load(config["train"]["ckpt_path"], map_location="cuda")
    dit = build_model("dit", dit_ckpt["config"]).cuda()
    dit.load_state_dict(dit_ckpt["model"])
    dit.eval()

    diffusion = GaussianDiffusion(device="cuda", **dit_ckpt["diffusion_config"])

    sample_cfg = config["sample"]
    n = sample_cfg["num_samples"]
    grid_size = dit_ckpt["config"]["grid_size"]
    latent_dim = dit_ckpt["config"]["latent_dim"]
    shape = (n, grid_size * grid_size, latent_dim)
    y = torch.full((n,), sample_cfg["class_id"], dtype=torch.long).cuda()

    # denoising latent noise
    z0 = diffusion.ddim_sample(dit, shape, y, steps=sample_cfg["ddim_steps"], device="cuda")

    # decode latents into RGB images
    with torch.no_grad():
        images = vae.decode(z0)


    # un-normalize 
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)

    images = images * std + mean
    out_path = sample_cfg["out_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(images, out_path, nrow=n)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/dit.yaml")
