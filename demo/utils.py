from typing import Optional, Tuple
import io

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from model import DEVICE, MODEL_PATH, load_model


test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

def load_random_sample() -> Image.Image:
    idx = np.random.randint(0, len(test_data))
    img = test_data[idx][0][0].numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")

# -------------------- preprocessing --------------------

def preprocess_canvas(canvas_data) -> Optional[torch.Tensor]:
    """Convert Sketchpad / PIL input -> 1x1x28x28 float32 tensor in [0,1].
    - Accepts PIL.Image or Gradio Sketchpad dict with 'image' or 'composite'.
    - Handles RGBA alpha by compositing on white.
    """
    if canvas_data is None:
        return None

    img = None
    if isinstance(canvas_data, dict):
        img = canvas_data.get("image") or canvas_data.get("composite")
    elif hasattr(canvas_data, "mode"):
        img = canvas_data

    if img is None:
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    elif img.mode != "L":
        img = img.convert("L")

    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0

    # revert to MNIST convention (white digit on black background)
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return t

# -------------------- visualization helpers --------------------

# def show_overlay(img_arr: np.ndarray, saliency: np.ndarray, cmap: str = "seismic") -> Image.Image: # try also "hot", ...
#     fig, ax = plt.subplots(figsize=(4, 4))
#     ax.imshow(img_arr, cmap="gray", vmin=0, vmax=1)
#     ax.imshow(saliency, cmap=cmap, alpha=0.6)#, interpolation="bilinear")
#     ax.axis("off")
#     plt.tight_layout(pad=0)
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
#     plt.close(fig)
#     buf.seek(0)
#     return Image.open(buf).convert("RGB")

def show_overlay(img_arr: np.ndarray, saliency: np.ndarray, cmap: str = "seismic", interpolation: str = "bilinear") -> Image.Image:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_arr, cmap="gray", vmin=0, vmax=1)
    ax.imshow(saliency, cmap=cmap, alpha=0.6, interpolation=interpolation)
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def plot_probabilities(probs: np.ndarray, pred: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    digits = list(range(len(probs)))
    colors = ["#ff4444" if i == pred else "#4444ff" for i in digits]
    bars = ax.bar(digits, probs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6)
    bars[pred].set_color("#ff4444")
    bars[pred].set_alpha(1.0)
    ax.set_xticks(digits)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_title("Class Probabilities")
    ax.set_xlabel("Class")
    ax.set_ylabel("Probability")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    for i, p in enumerate(probs):
        if p > 0.01:
            ax.text(i, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# -------------------- predict wrapper --------------------

def predict_tensor(x: torch.Tensor) -> Tuple[int, np.ndarray]:
    """Return (pred_label, probs[10]). Use model on DEVICE."""
    model = load_model(MODEL_PATH)
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs