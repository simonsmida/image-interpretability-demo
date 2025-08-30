---
title: Interpretability Methods Throughout the Years (MNIST)
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.44.1
app_file: demo/app.py
pinned: false
---

# Interpretability Methods Throughout the Years (with MNIST)

Draw a digit or load a random MNIST sample, select an interpretability method, and view the saliency overlay. This project follows [this article](https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/) and demonstrates various neural network interpretation techniques on handwritten digits.

## Features
- Interactive digit drawing canvas
- Multiple interpretability methods (Vanilla Gradient, and others)
- Real-time saliency visualization
- Probability distribution display
- Smooth overlay options

## Usage
1. Draw a digit on the canvas or load a random MNIST sample
2. Select an interpretability method from the dropdown
3. Choose target class (auto-detection or manual selection)
4. View the saliency overlay showing what the model focuses on