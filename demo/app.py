"""
Gradio app following this article: https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/
Run with: python app.py
"""

import io
import os
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F

import gradio as gr
import matplotlib.pyplot as plt

from model import SmallCNN

# -------------------- configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mnist_model.pth"


_model = None

def load_model(path: Optional[str] = None):
    global _model
    if _model is not None:
        return _model
    model = SmallCNN().to(DEVICE)
    if path and os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            print("Loaded model from", path)
        except Exception as e:
            print("Failed to load model:", e)
    model.eval()
    _model = model
    return _model

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

# -------------------- predict wrapper --------------------

def predict_tensor(x: torch.Tensor) -> Tuple[int, np.ndarray]:
    """Return (pred_label, probs[10]). Use model on DEVICE."""
    model = load_model(MODEL_PATH)
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs

# -------------------- visualization helpers --------------------

def show_overlay(img_arr: np.ndarray, saliency: np.ndarray, cmap: str = "seismic") -> Image.Image: # try also "hot", ...
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_arr, cmap="gray", vmin=0, vmax=1)
    ax.imshow(saliency, cmap=cmap, alpha=0.6, interpolation="bilinear")
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
    bars = ax.bar(digits, probs, alpha=0.8, edgecolor="black", linewidth=0.6)
    ax.set_xticks(digits)
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

# -------------------- interpretability methods (TODO expand) --------------------

def null_saliency(input_tensor: torch.Tensor, target: Optional[int] = None) -> np.ndarray:
    """Return a flat zero saliency. Replace with actual methods."""
    return np.zeros((28, 28), dtype=np.float32)

# TODO: implement vanilla_gradient, smoothgrad, integrated_gradients, grad_cam, etc.

METHODS = {
    "Null (placeholder)": null_saliency,
    # add method names -> functions here
}

# -------------------- core app handler --------------------

def run_app(canvas_data, 
            smooth_noise: float,  blur_sigma: float,
            sample_digit: str, method_name: str, target_class: str,
            loo_patch: int, smooth_n: int, ig_steps: int, blur_steps: int):
    # input selection
    if canvas_data is not None:
        t = preprocess_canvas(canvas_data)
        if t is None:
            # fallback: random sample path
            return None, gr.update(label="No valid input"), None
    else: # fallback sample: blank image
        t = torch.zeros(1, 1, 28, 28)

    pred, probs = predict_tensor(t)
    pred_str = f"Predicted: {pred} (conf {probs[pred]:.3f})"

    # method selection and call method
    fn = METHODS.get(method_name, null_saliency)

    # skeleton methods currently ignore extra params; adapt per-method
    sal = fn(t, target=None)

    img_arr = t[0, 0].cpu().numpy()
    overlay = show_overlay(img_arr, sal)
    prob_plot = plot_probabilities(probs, pred)
    return overlay, gr.update(label=pred_str), prob_plot

# -------------------- UI --------------------

def build_ui():
    title_str = "Interpretability Methods Throughout the Years (with MNIST)"
    with gr.Blocks(title=title_str) as demo:
        gr.Markdown(f"## {title_str}")
        gr.Markdown("<br>")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input")
                        canvas = gr.ImageEditor(
                            label="Draw digit",
                            type="pil",
                            image_mode="L",   # grayscale
                            height=300,
                            brush=gr.Brush(default_size=30, colors=["#000000"], color_mode="fixed"),
                            eraser=gr.Eraser(default_size=40),
                            transforms=None,  # disable crop/resize
                            layers=False,     # disable multiple layers
                            # show_download_button=False,
                            # show_fullscreen_button=False,
                        )
                        gr.Markdown("<br><br>")
                        gr.Markdown("Draw a digit (0-9) or use a test sample, select an interpretability method, and see the saliency overlay.")
                        gr.Markdown("<br>")
                        gr.Markdown("This project is following [this article](https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/) and can be found on [my GitHub](www.simonsmida.github.io/).")
                    with gr.Column():
                        gr.Markdown("<br>")
                        sample_digit = gr.Dropdown(choices=["random"] + [str(i) for i in range(10)], value="random", label="Sample")
                        sample_btn = gr.Button("New sample")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Interpretability Method")
                        method = gr.Dropdown(list(METHODS.keys()), value="Null (placeholder)", label="Method Selection")
                        target = gr.Dropdown(["auto"] + [str(i) for i in range(10)], value="auto", label="Target Class")
                        run_btn = gr.Button("Run", variant="primary")
                        pred_group = gr.Markdown("Model Prediction: —")
                    with gr.Column():
                        gr.Markdown("### Output")
                        out_img = gr.Image(type="pil", label="Overlay")
                        prob_plot = gr.Image(type="pil", show_label=False)#label="Class Probs")

        inputs_all = [canvas, sample_digit, method, target]
        run_btn.click(run_app, inputs=inputs_all, outputs=[out_img, pred_group, prob_plot])
        sample_btn.click(run_app, inputs=inputs_all, outputs=[out_img, pred_group, prob_plot])

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(debug=True, share=False)