"""Neural style transfer: the optimization loop, saving, and run records.

Command line:
    python src/stylize.py --content data/content/384/seaview.jpg \
                          --style data/style/384/great_wave.jpg

Two ways to use it from code:
    run(...)                 runs to the end, saves the image + JSON, returns paths
    for p in stylize(...):   yields a Progress every `preview_every` iterations
                             (this is what the Gradio app will use for FR5)
"""

import argparse
import json
import platform
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch

from features import VGGFeatures, CONTENT_LAYERS, STYLE_LAYERS
from images import load_pair, deprocess, IMAGENET_MEAN, IMAGENET_STD, DEFAULT_SIZE
from losses import content_loss, gram_matrix, style_loss, tv_loss

DEFAULTS = dict(
    size=DEFAULT_SIZE,
    iterations=200,
    content_weight=1.0,
    style_weight=1e6,
    tv_weight=1e-2,     # starting point; tune with real weights in Phase 4
    init="content",     # "content" or "noise"
    seed=0,
)


@dataclass
class Progress:
    iteration: int
    iterations: int
    content: float
    style: float
    tv: float
    total: float
    image: torch.Tensor          # current image, normalized (call deprocess to view)
    history: list = field(default_factory=list)  # [(content, style, tv, total), ...]
    done: bool = False
    cancelled: bool = False


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _valid_range(device):
    """Normalized values that correspond to pixel values 0 and 1, per channel."""
    lo = ((0 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
    hi = ((1 - IMAGENET_MEAN) / IMAGENET_STD).to(device)
    return lo, hi


def stylize(content_path, style_path, *, size=DEFAULTS["size"], iterations=DEFAULTS["iterations"],
            content_weight=DEFAULTS["content_weight"], style_weight=DEFAULTS["style_weight"],
            tv_weight=DEFAULTS["tv_weight"], init=DEFAULTS["init"], seed=DEFAULTS["seed"],
            style_layers=STYLE_LAYERS, content_layers=CONTENT_LAYERS, preview_every=10,
            device="auto", stop_event: threading.Event | None = None, net=None, pretrained=True):
    """Generator. Yields a Progress every `preview_every` iterations and at the end.

    Set `stop_event` (or simply stop iterating) to cancel; the last yield then
    has cancelled=True and holds the most recent image.
    """
    device = pick_device(device)
    torch.manual_seed(seed)

    content, style = load_pair(content_path, style_path, size, device)
    net = net or VGGFeatures(list(content_layers) + list(style_layers), pretrained=pretrained, device=device)

    # Targets: computed once, never change
    with torch.no_grad():
        content_targets = {l: f for l, f in net(content).items() if l in content_layers}
        style_targets = {l: gram_matrix(f) for l, f in net(style).items() if l in style_layers}

    # The image we optimize
    if init == "content":
        img = content.clone()
    elif init == "noise":
        img = torch.randn_like(content)
    else:
        raise ValueError("init must be 'content' or 'noise'")
    img.requires_grad_(True)

    # max_iter=1: one step per call, so we can report between steps.
    # Tolerances 0: by default LBFGS quietly stops updating once gradients get tiny;
    # we want exactly the number of iterations the user asked for.
    optimizer = torch.optim.LBFGS([img], max_iter=1, tolerance_grad=0.0, tolerance_change=0.0)
    lo, hi = _valid_range(device)
    history = []
    last = {}

    def closure():
        optimizer.zero_grad()
        feats = net(img)
        c = sum(content_loss(feats[l], content_targets[l]) for l in content_layers) if content_weight else torch.zeros((), device=device)
        s = style_loss(feats, style_targets) if style_weight else torch.zeros((), device=device)
        t = tv_loss(img) if tv_weight else torch.zeros((), device=device)
        total = content_weight * c + style_weight * s + tv_weight * t
        total.backward()
        last.update(content=c.item(), style=s.item(), tv=t.item(), total=total.item())
        return total

    def report(i, done=False, cancelled=False):
        return Progress(i, iterations, last["content"], last["style"], last["tv"], last["total"],
                        img.detach().clone(), history, done, cancelled)

    for i in range(1, iterations + 1):
        optimizer.step(closure)
        with torch.no_grad():  # keep pixels inside the displayable range
            img.copy_(torch.max(torch.min(img, hi), lo))
        history.append((last["content"], last["style"], last["tv"], last["total"]))

        if stop_event is not None and stop_event.is_set():
            yield report(i, done=True, cancelled=True)
            return
        if i == iterations:
            yield report(i, done=True)
        elif i % preview_every == 0:
            yield report(i)


def save_result(progress: Progress, settings: dict, runtime_s: float, out_dir="outputs", fmt="png"):
    """Save the image and its run record (FR8). Returns (image_path, json_path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{Path(settings['content']).stem}__{Path(settings['style']).stem}__{stamp}"

    image = deprocess(progress.image)
    img_path = out_dir / f"{stem}.{fmt}"
    image.save(img_path, quality=95) if fmt in ("jpg", "jpeg") else image.save(img_path)

    record = {
        "output": img_path.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "content_image": Path(settings["content"]).name,
        "style_image": Path(settings["style"]).name,
        "output_size": list(image.size),
        "settings": {k: settings[k] for k in
                     ("size", "iterations", "content_weight", "style_weight", "tv_weight", "init", "seed")},
        "layers": {"content": list(settings["content_layers"]), "style": list(settings["style_layers"])},
        "iterations_completed": progress.iteration,
        "cancelled": progress.cancelled,
        "final_loss": {"content": progress.content, "style": progress.style,
                       "tv": progress.tv, "total": progress.total},
        "runtime_seconds": round(runtime_s, 2),
        "environment": {"device": settings["device"], "torch": torch.__version__,
                        "python": platform.python_version(), "machine": platform.platform()},
        "loss_history": [dict(zip(("content", "style", "tv", "total"), h)) for h in progress.history],
    }
    json_path = img_path.with_suffix(".json")
    json_path.write_text(json.dumps(record, indent=2))
    return img_path, json_path


def run(content_path, style_path, out_dir="outputs", fmt="png", verbose=True, **kwargs):
    """Run a full style transfer, save the result and its record, return the paths."""
    settings = {**DEFAULTS, **kwargs, "content": str(content_path), "style": str(style_path),
                "content_layers": kwargs.get("content_layers", CONTENT_LAYERS),
                "style_layers": kwargs.get("style_layers", STYLE_LAYERS),
                "device": pick_device(kwargs.get("device", "auto"))}
    start = time.time()
    last = None
    for p in stylize(content_path, style_path, **kwargs):
        last = p
        if verbose:
            print(f"iter {p.iteration:4d}/{p.iterations}  total {p.total:10.3f}  "
                  f"content {p.content:8.3f}  style {p.style:.6f}  tv {p.tv:.4f}")
    return save_result(last, settings, time.time() - start, out_dir, fmt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Neural style transfer (Gatys et al.)")
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--size", type=int, default=DEFAULTS["size"])
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    ap.add_argument("--style-weight", type=float, default=DEFAULTS["style_weight"])
    ap.add_argument("--content-weight", type=float, default=DEFAULTS["content_weight"])
    ap.add_argument("--tv-weight", type=float, default=DEFAULTS["tv_weight"])
    ap.add_argument("--init", choices=["content", "noise"], default=DEFAULTS["init"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    ap.add_argument("--untrained", action="store_true", help="offline testing only")
    a = ap.parse_args()

    img_path, json_path = run(a.content, a.style, out_dir=a.out, fmt=a.format,
                              size=a.size, iterations=a.iterations, style_weight=a.style_weight,
                              content_weight=a.content_weight, tv_weight=a.tv_weight,
                              init=a.init, seed=a.seed, device=a.device, pretrained=not a.untrained)
    print(f"Saved {img_path}\nSaved {json_path}")
