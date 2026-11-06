import time, torch, torchvision, gradio, PIL, matplotlib
from torchvision.models import vgg19, VGG19_Weights

print("torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| gradio", gradio.__version__, "| pillow", PIL.__version__)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# First run downloads about 550 MB, then it's cached
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29].eval().to(device)  # up to conv5_1
for p in vgg.parameters():
    p.requires_grad_(False)

x = torch.rand(1, 3, 256, 256, device=device, requires_grad=True)
t = time.time()
for _ in range(5):
    vgg(x).sum().backward()          # forward plus backward, like one optimization step
print(f"avg time per step at 256 px: {(time.time() - t) / 5:.2f} s")
print("VGG-19 loaded OK")