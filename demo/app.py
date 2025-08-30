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
from utils import show_overlay, plot_probabilities, preprocess_canvas, predict_tensor, load_random_sample
from methods import METHODS


# -------------------- core app handler --------------------
def run_app(canvas_data, method_name: str, target_class: str, smooth_overlay: bool):
    if canvas_data is not None:
        t = preprocess_canvas(canvas_data)
        if t is None:
            return None, gr.update(label="No valid input"), None
    else:
        t = torch.zeros(1, 1, 28, 28)

    pred, probs = predict_tensor(t)
    pred_str = f"Prediction: {pred} (conf {probs[pred]:.3f})"
    print("Prediction:", pred_str)
    fn = METHODS.get(method_name)
    print("Using method:", method_name)
    if target_class == "auto":
        target = None
    else:
        target = int(target_class)  # Convert string to int

    sal = fn(t, target=target)

    img_arr = t[0, 0].cpu().detach().numpy()

    interp = "bilinear" if smooth_overlay else "nearest"
    overlay = show_overlay(img_arr, sal, interpolation=interp)
    prob_plot = plot_probabilities(probs, pred)
    return overlay, gr.update(label=pred_str), prob_plot

# -------------------- UI --------------------

def build_ui():
    title_str = "Interpretability Methods Throughout the Years (with MNIST)"
    with gr.Blocks(title=title_str, css="""
        .arrow {
            font-size: 48px; 
            text-align: center; 
            line-height: 300px; 
            color: #444;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 300px;
        }
        .arrow-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }
    """) as demo:
        gr.Markdown(f"## {title_str}")
        gr.Markdown("Draw a digit or load a random MNIST sample, select an interpretability method, and view the saliency overlay. This project follows [this article](https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/) and is on [my GitHub](https://simonsmida.github.io/).")
        gr.Markdown("<br>")
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=3):
                gr.Markdown("### Input")
                
                # Create initial blank image
                initial_image = Image.new('L', (280, 280), 255)
                
                canvas = gr.ImageEditor(
                    label="Draw digit or load a sample",
                    type="pil",
                    image_mode="L",
                    height=300,
                    value=initial_image,
                    brush=gr.Brush(default_size=15, colors=["#000000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=20),
                    transforms=None,
                    layers=False,
                )
                with gr.Row():
                    load_sample_btn = gr.Button("Load Random Digit")
                    clear_btn = gr.Button("Clear Canvas")

            # Arrow 1
            with gr.Column(scale=1, min_width=60):
                gr.HTML('<div class="arrow-container"><div class="arrow">→</div></div>')

            # Interpretability Method Column
            with gr.Column(scale=3):
                gr.Markdown("### Interpretability Method")
                method = gr.Dropdown(list(METHODS.keys()), value="Vanilla Gradient", label="Method Selection")
                target = gr.Dropdown(["auto"] + [str(i) for i in range(10)], value="auto", label="Target Class")
                run_btn = gr.Button("Run", variant="primary")
                with gr.Accordion(label="Predicted: —", open=False) as pred_group:
                    prob_plot = gr.Image(type="pil", show_label=False, height=150)

            # Arrow 2
            with gr.Column(scale=1, min_width=60):
                gr.HTML('<div class="arrow-container"><div class="arrow">→</div></div>')

            # Output Column
            with gr.Column(scale=3):
                gr.Markdown("### Output")
                out_img = gr.Image(type="pil", label="Overlay", height=300)
                interpolation_toggle = gr.Checkbox(label="Smooth saliency overlay (bilinear)", value=True)

        def clear_canvas():
            return Image.new('L', (280, 280), 255)

        # Define all inputs
        inputs_all = [canvas, method, target, interpolation_toggle]
        outputs_all = [out_img, pred_group, prob_plot]

        # Manual run button (keep this for explicit runs)
        run_btn.click(run_app, inputs=inputs_all, outputs=outputs_all)
        
        # Auto-update on parameter changes
        canvas.change(run_app, inputs=inputs_all, outputs=outputs_all)
        method.change(run_app, inputs=inputs_all, outputs=outputs_all)
        target.change(run_app, inputs=inputs_all, outputs=outputs_all)
        interpolation_toggle.change(run_app, inputs=inputs_all, outputs=outputs_all)
        
        # Auto-update when canvas changes (optional - might be too frequent)
        # canvas.change(run_app, inputs=inputs_all, outputs=outputs_all)
        
        # Button actions
        load_sample_btn.click(load_random_sample, inputs=[], outputs=[canvas])
        clear_btn.click(clear_canvas, inputs=[], outputs=[canvas])
        
        # Auto-update after loading a sample
        load_sample_btn.click(run_app, inputs=inputs_all, outputs=outputs_all)

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(debug=True, share=False)