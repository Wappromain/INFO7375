"""Sanity check for the image and feature extractor steps.

Run from the repo root:  python src/check_steps.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from images import load_pair, deprocess
from features import VGGFeatures, CONTENT_LAYERS, STYLE_LAYERS

PRETRAINED = "--untrained" not in sys.argv  # --untrained only for offline testing

content, style = load_pair("data/content/384/seaview.jpg", "data/style/384/great_wave.jpg", size=256)
print("content tensor:", tuple(content.shape), "| style tensor:", tuple(style.shape))

# 1. preprocess -> deprocess should give back the same picture
back = deprocess(content)
print("round trip size:", back.size, "(should match content W x H)")

# 2. feature extractor
net = VGGFeatures(CONTENT_LAYERS + STYLE_LAYERS, pretrained=PRETRAINED)
print("modules kept:", len(net.body), "(cut after conv5_1)")
print("frozen:", all(not p.requires_grad for p in net.parameters()))

feats = net(content)
for name in STYLE_LAYERS + CONTENT_LAYERS:
    print(f"  {name:8s} {tuple(feats[name].shape)}")

# 3. gradients must reach the image, not the network
img = content.clone().requires_grad_(True)
net(img)["conv4_2"].mean().backward()
print("gradient reaches image:", img.grad is not None and img.grad.abs().sum().item() > 0)
print("All checks passed.")
