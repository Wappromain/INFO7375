"""The three loss terms.

total = content_weight * content_loss
      + style_weight   * style_loss
      + tv_weight      * tv_loss
"""

import torch
import torch.nn.functional as F


def content_loss(feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """How different our image's conv4_2 features are from the photo's.

    Low = the objects and layout of the photo are still there.
    """
    return F.mse_loss(feat, target)


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Which filters fire together, ignoring where.

    feat: (1, C, H, W) -> (C, C).  Entry [i, j] is the dot product of filter i's
    map with filter j's map, summed over every position. Summing over positions
    throws away layout and keeps only texture and color statistics.

    Divided by C*H*W so layers of different sizes contribute on similar scales.
    """
    _, c, h, w = feat.shape
    f = feat.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


def style_loss(feats: dict, target_grams: dict, layer_weights: dict | None = None) -> torch.Tensor:
    """Sum over style layers of the difference between Gram matrices.

    target_grams: precomputed Gram matrices of the painting, {layer: (C, C)}.
    layer_weights: optional {layer: weight}; equal weights by default.
    """
    total = 0.0
    for layer, target in target_grams.items():
        w = 1.0 if layer_weights is None else layer_weights.get(layer, 0.0)
        if w:
            total = total + w * F.mse_loss(gram_matrix(feats[layer]), target)
    return total


def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """Total variation: how much each pixel differs from its right and lower neighbor.

    Penalizing it removes high frequency speckle noise that style transfer
    tends to create, without blurring real edges much.
    """
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx.pow(2).mean() + dy.pow(2).mean()
