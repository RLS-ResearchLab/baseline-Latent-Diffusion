import os
import sys
import yaml
import torch

from models import build_model
from losses import vae_loss
from data import get_dataloader


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vae = build_model("vae", config["model"]).cuda()
    vae.compile()
    opt = torch.optim.AdamW(vae.parameters(), lr=config["train"]["lr"])
    loader = get_dataloader(config["data"])

    amp_enabled = config["train"]["use_amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    step = 0
    for epoch in range(config["train"]["epochs"]):
        for x, _ in loader:
            x = x.to(device)

            opt.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                recon, mu, logvar = vae(x)
                loss, recon_loss, kl = vae_loss(
                    recon,
                    x,
                    mu,
                    logvar,
                    config["train"]["kl_weight"],
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % config["train"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"recon {recon_loss.item():.4f} kl {kl.item():.4f}")
            step += 1

    ckpt_path = config["train"]["ckpt_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": vae.state_dict(), "config": config["model"]}, ckpt_path)
    print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "vae.yaml")