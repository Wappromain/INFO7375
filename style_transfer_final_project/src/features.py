"""Frozen VGG-19 feature extractor.

We never train VGG-19. We only read its activations at a few named layers:
  * content: conv4_2
  * style:   conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

Naming: torchvision stores VGG-19 as one flat list (`vgg19().features`) with
index numbers. The table below gives those indices their usual names. Like
Gatys et al., we take each layer's output after its ReLU, so "conv4_2" means
the output of ReLU at index 22.

The network is cut right after the deepest layer we ask for, so nothing
unused is ever computed (E3 asks for conv5_2, so it cuts a little later).
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

CONTENT_LAYERS = ["conv4_2"]
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]


def _vgg19_layer_names():
    """Name every module in vgg19().features, e.g. index 21 -> 'conv4_2'."""
    names, block, conv = [], 1, 0
    for convs_in_block in [2, 2, 4, 4, 4]:
        for _ in range(convs_in_block):
            conv += 1
            names += [f"conv{block}_{conv}_raw", f"conv{block}_{conv}"]  # conv, then its ReLU
        names.append(f"pool{block}")
        block, conv = block + 1, 0
    return names


LAYER_NAMES = _vgg19_layer_names()  # 37 names, one per module


class VGGFeatures(nn.Module):
    """Returns {layer_name: activation} for the requested layers.

    Example:
        net = VGGFeatures(CONTENT_LAYERS + STYLE_LAYERS)
        feats = net(x)          # feats["conv4_2"].shape == (1, 512, H/8, W/8)
    """

    def __init__(self, layers, avg_pool: bool = False, pretrained: bool = True, device: str = "cpu"):
        super().__init__()
        unknown = [l for l in layers if l not in LAYER_NAMES]
        if unknown:
            raise ValueError(f"Unknown layer(s) {unknown}. Valid: conv1_1 ... conv5_4")

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        full = vgg19(weights=weights).features

        last = max(LAYER_NAMES.index(l) for l in layers)
        modules = list(full.children())[: last + 1]  # cut after the deepest layer we need
        if avg_pool:  # optional: Gatys et al. report slightly smoother results
            modules = [nn.AvgPool2d(2, 2) if isinstance(m, nn.MaxPool2d) else m for m in modules]
        # inplace ReLU would overwrite activations we want to keep
        modules = [nn.ReLU(inplace=False) if isinstance(m, nn.ReLU) else m for m in modules]

        self.body = nn.Sequential(*modules).to(device).eval()
        self.layers = set(layers)
        for p in self.body.parameters():
            p.requires_grad_(False)  # frozen: we optimize the image, not the network

    def forward(self, x):
        out = {}
        for i, module in enumerate(self.body):
            x = module(x)
            name = LAYER_NAMES[i]
            if name in self.layers:
                out[name] = x
        return out
