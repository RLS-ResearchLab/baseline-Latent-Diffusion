import os
import sys
import yaml
import torch

from src.models import build_model
from src.losses import diffusion_loss
from src.diffusion import GaussianDiffusion
from src.data import get_dataloader


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vae_ckpt = torch.load(config["train"]["vae_ckpt_path"], map_location="cuda")
    vae = build_model("vae", vae_ckpt["config"]).cuda()
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    dit = build_model("dit", config["model"]).cuda()

    vae.compile()
    dit.compile()

    diffusion = GaussianDiffusion(device="cuda", **config["diffusion"])
    opt = torch.optim.AdamW(dit.parameters(), lr=config["train"]["lr"])
    loader = get_dataloader(config["data"])

    amp_enabled = config["train"]["use_amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    step = 0
    for epoch in range(config["train"]["epochs"]):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                with torch.no_grad():
                    z0 = vae.encode(x, sample=False)  # (B, N, latent_dim)

                loss = diffusion_loss(dit, diffusion, z0, y, train=True)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % config["train"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

    ckpt_path = config["train"]["ckpt_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": dit.state_dict(), "config": config["model"],
                "diffusion_config": config["diffusion"]}, ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config/dit.yaml")