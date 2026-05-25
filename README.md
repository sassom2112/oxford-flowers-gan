# Adversarial Image Synthesis — Oxford 102 Flowers

A DCGAN trained to synthesize photorealistic flower images, with analysis of the adversarial training dynamics between Generator and Discriminator.

---

## Why This Project

GANs are the original adversarial ML framework. Unlike classification-focused adversarial work (e.g., FGSM on a trained model), here the adversarial game *is* the training loop — the Discriminator learns to detect fakes while the Generator learns to defeat it. Studying their loss dynamics reveals the same instability patterns that make GANs exploitable in real-world deepfake detection.

---

## Architecture

**DCGAN** — Radford et al. (2015), trained on the Oxford 102 Flowers dataset (1,020 training images, 64×64 RGB).

| Component | Design |
|---|---|
| Generator | ConvTranspose2d × 5, BatchNorm, ReLU → Tanh, input: z ∈ ℝ¹⁰⁰ |
| Discriminator | Conv2d × 5, BatchNorm, LeakyReLU(0.2) → Sigmoid |
| Loss | Binary Cross-Entropy (adversarial) |
| Optimizer | Adam, lr=0.0001, β₁=0.5 |
| Epochs | 250 |

Custom weight initialization (Conv: N(0, 0.02), BatchNorm: N(1, 0.02)) per the original paper.

---

## Training Dynamics

The adversarial instability is visible in the loss curves:

| Epoch | D Loss | G Loss | D(x) | D(G(z)) |
|---|---|---|---|---|
| 1 | 1.5698 | 2.9768 | 0.517 | 0.490 → 0.072 |
| 5 | 0.1720 | 11.384 | 0.865 | 0.001 → 0.000 |
| 15 | 0.1729 | 6.041 | 0.934 | 0.030 → 0.003 |

**Key observations:**
- The Discriminator converges fast and dominates early training — G Loss spikes above 10 by epoch 5, indicating the Generator is being crushed before it learns useful structure.
- Recovery by epoch 15 (G Loss drops to ~6) shows the Generator adapting, but D(G(z)) remaining near zero suggests mode collapse risk.
- The `β₁=0.5` (vs. default 0.9) is critical for GAN stability — higher momentum causes oscillation in adversarial gradients.

---

## Results

Generated images are saved per epoch in `generated_images/`. Loss curves for G and D are plotted at the end of training.

---

## Stack

`PyTorch` · `torchvision` · `NumPy` · `Matplotlib`

---

## Extensions

- **Conditional GAN (cGAN)** — condition on flower species label to control synthesis
- **FID metric** — quantitative evaluation of image quality vs. real distribution
- **Discriminator as detector** — repurpose the trained D as a deepfake/synthetic image classifier, connecting GAN training directly to adversarial forensics
