import os
import sys
import yaml
import torch
from torchvision.utils import save_image

from src.models import build_model
from src.data import get_dataloader


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ckpt = torch.load(config["train"]["ckpt_path"], map_location="cuda")
    vae = build_model("vae", ckpt["config"]).cuda()
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    loader = get_dataloader(config["data"])
    x, _ = next(iter(loader))
    x = x[:8].cuda()

    with torch.no_grad():
        recon, _, _ = vae(x)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

    grid = torch.cat([x, recon], dim=0) * std + mean
    out_path = "samples/vae_recon.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path, nrow=x.shape[0])
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/vae.yaml")
