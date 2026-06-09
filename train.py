"""
SageMaker training entry point for Oxford Flowers DCGAN.

SageMaker injects these env vars automatically:
  SM_MODEL_DIR        -> /opt/ml/model        (artifacts saved here go to S3)
  SM_OUTPUT_DATA_DIR  -> /opt/ml/output/data  (generated images, plots)
  SM_CHANNEL_TRAINING -> /opt/ml/input/data/training  (optional: pre-staged S3 data)
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Hyperparameters
    p.add_argument("--epochs",      type=int,   default=250)
    p.add_argument("--lr",          type=float, default=0.0001)
    p.add_argument("--beta1",       type=float, default=0.5)
    p.add_argument("--batch-size",  type=int,   default=102)
    p.add_argument("--nz",          type=int,   default=100)
    p.add_argument("--ngf",         type=int,   default=64)
    p.add_argument("--ndf",         type=int,   default=64)
    p.add_argument("--image-size",  type=int,   default=64)
    p.add_argument("--log-interval",type=int,   default=25,
                   help="Print metrics every N epochs")

    # SageMaker directories (fall back to local paths for testing)
    p.add_argument("--model-dir",
                   default=os.environ.get("SM_MODEL_DIR", "model"))
    p.add_argument("--output-data-dir",
                   default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    p.add_argument("--data-dir",
                   default=os.environ.get("SM_CHANNEL_TRAINING", ""))

    return p.parse_args()


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    cls = m.__class__.__name__
    if cls.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cls.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # If a pre-staged data channel is provided (S3 → container), use it.
    # Otherwise download directly (Colab / local dev).
    root = args.data_dir if args.data_dir else ""
    dataset = torchvision.datasets.Flowers102(
        root, split="train", transform=transform, download=True
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args, device):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)
    img_dir = os.path.join(args.output_data_dir, "epoch_images")
    os.makedirs(img_dir, exist_ok=True)

    loader = get_dataloader(args)
    print(f"Dataset: {len(loader.dataset)} images, {len(loader)} batches/epoch")

    netD = Discriminator(nc=3, ndf=args.ndf).to(device)
    netG = Generator(nc=3, nz=args.nz, ngf=args.ngf).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)

    criterion  = nn.BCELoss()
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    history = {"epoch": [], "D_loss": [], "G_loss": [], "D_x": [], "D_Gz": []}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_d, epoch_g, epoch_dx, epoch_dgz = [], [], [], []

        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            b = real_imgs.size(0)

            # Discriminator
            optimizerD.zero_grad()
            out_real = netD(real_imgs).view(-1)
            loss_d_real = criterion(out_real, torch.ones(b, device=device))
            loss_d_real.backward()

            noise = torch.randn(b, args.nz, 1, 1, device=device)
            fake = netG(noise)
            out_fake = netD(fake.detach()).view(-1)
            loss_d_fake = criterion(out_fake, torch.zeros(b, device=device))
            loss_d_fake.backward()
            optimizerD.step()

            # Generator
            optimizerG.zero_grad()
            out_fake2 = netD(fake).view(-1)
            loss_g = criterion(out_fake2, torch.ones(b, device=device))
            loss_g.backward()
            optimizerG.step()

            epoch_d.append((loss_d_real + loss_d_fake).item())
            epoch_g.append(loss_g.item())
            epoch_dx.append(out_real.mean().item())
            epoch_dgz.append(out_fake2.mean().item())

        d_loss = np.mean(epoch_d)
        g_loss = np.mean(epoch_g)
        d_x    = np.mean(epoch_dx)
        d_gz   = np.mean(epoch_dgz)

        history["epoch"].append(epoch)
        history["D_loss"].append(d_loss)
        history["G_loss"].append(g_loss)
        history["D_x"].append(d_x)
        history["D_Gz"].append(d_gz)

        # Emit SageMaker-parseable metric lines
        print(f"epoch={epoch}; D_loss={d_loss:.4f}; G_loss={g_loss:.4f}; "
              f"D_x={d_x:.4f}; D_Gz={d_gz:.4f};")

        if epoch % args.log_interval == 0 or epoch == 1:
            elapsed = (time.time() - t0) / 60
            print(f"  [{elapsed:.1f}m] Epoch {epoch}/{args.epochs}")

        # Save checkpoint image
        with torch.no_grad():
            fake_fixed = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake_fixed, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
        plt.savefig(os.path.join(img_dir, f"epoch_{epoch:04d}.png"),
                    bbox_inches="tight", dpi=100)
        plt.close()

    return netD, netG, history


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(netD, netG, history, args):
    torch.save(netG.state_dict(), os.path.join(args.model_dir, "generator.pth"))
    torch.save(netD.state_dict(), os.path.join(args.model_dir, "discriminator.pth"))

    with open(os.path.join(args.output_data_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Loss curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(history["epoch"], history["G_loss"], color="#e74c3c", label="Generator")
    axes[0].set_ylabel("G Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["D_loss"], color="#3498db", label="Discriminator")
    axes[1].set_ylabel("D Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle("Adversarial Training Dynamics", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_data_dir, "training_dynamics.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Artifacts saved → model: {args.model_dir}, outputs: {args.output_data_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, "
          f"batch_size={args.batch_size}, nz={args.nz}")

    netD, netG, history = train(args, device)
    save_artifacts(netD, netG, history, args)
