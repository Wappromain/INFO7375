"""Loading, preparing and saving images for VGG-19.

VGG-19 was trained on ImageNet photos that were:
  * RGB, values scaled to [0, 1]
  * normalized per channel with the ImageNet mean and std below
So every image we feed it must go through the same steps (preprocess),
and every result we show must have them undone (deprocess).
"""

from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
import torchvision.transforms.functional as TF

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

ALLOWED_FORMATS = {"JPEG", "PNG"}
DEFAULT_SIZE = 256
MAX_SIZE = 384


class ImageInputError(ValueError):
    """Raised for files we refuse to process (FR2). The message is shown to the user."""


def load_image(path_or_file, max_size: int = DEFAULT_SIZE) -> Image.Image:
    """Open an image, check it is a valid JPG or PNG, and resize it.

    The longer side is scaled down to `max_size` (aspect ratio kept).
    Small images are never scaled up.
    """
    if max_size > MAX_SIZE:
        raise ImageInputError(f"Maximum working size is {MAX_SIZE} px, got {max_size}.")

    try:
        img = Image.open(path_or_file)
        fmt = img.format
        img.load()  # forces a full decode, so truncated or corrupt files fail here
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        name = Path(str(path_or_file)).name
        raise ImageInputError(f"'{name}' is not a readable image file.") from e

    if fmt not in ALLOWED_FORMATS:
        raise ImageInputError(f"Unsupported format '{fmt}'. Please upload a JPG or PNG.")

    img = ImageOps.exif_transpose(img)  # respect phone camera rotation
    img = img.convert("RGB")            # drops alpha (PNG) and handles grayscale
    img.thumbnail((max_size, max_size), Image.LANCZOS)  # shrinks only, keeps aspect
    return img


def preprocess(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """PIL image -> normalized tensor of shape (1, 3, H, W) ready for VGG-19."""
    x = TF.to_tensor(img).unsqueeze(0)            # [0, 1], shape (1, 3, H, W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.to(device)


def deprocess(x: torch.Tensor) -> Image.Image:
    """Normalized tensor (1, 3, H, W) -> displayable PIL image. Undoes preprocess."""
    x = x.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0, 1).squeeze(0)
    return TF.to_pil_image(x)


def load_pair(content_path, style_path, size: int = DEFAULT_SIZE, device: str = "cpu"):
    """Load a content and style image at the same working size.

    The style image is resized to the same longer side as the content image,
    so its brushstrokes appear at a comparable scale in the result.
    """
    content = load_image(content_path, size)
    style = load_image(style_path, size)
    return preprocess(content, device), preprocess(style, device)
