import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np

from model import get_model

def integrated_gradients(input_tensor, target=None, n_steps=50):
    """Manual Integrated Gradients implementation"""
    model = get_model()
    
    if target is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    # Create baseline (black image)
    baseline = torch.zeros_like(input_tensor)
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, n_steps)
    gradients = []
    
    for alpha in alphas:
        # Interpolate between baseline and input
        interpolated = baseline + alpha * (input_tensor - baseline)
        interpolated.requires_grad_(True)
        
        # Forward pass
        logits = model(interpolated)
        score = logits[0, target]
        
        # Backward pass
        score.backward()
        
        # Store gradient
        grad = interpolated.grad[0, 0].cpu().numpy()
        gradients.append(grad)
        
        # Clear gradients
        interpolated.grad = None
    
    # Average gradients and multiply by input difference
    avg_gradients = np.mean(gradients, axis=0)
    input_diff = (input_tensor - baseline)[0, 0].cpu().numpy()
    integrated_grad = avg_gradients * input_diff
    
    # Take absolute value and normalize
    integrated_grad = np.abs(integrated_grad)
    integrated_grad = integrated_grad - integrated_grad.min()
    if integrated_grad.max() > 0:
        integrated_grad = integrated_grad / integrated_grad.max()
    
    return integrated_grad

def integrated_gradients_captum(input_tensor, target=None, n_steps=50):
    """Integrated Gradients using Captum"""
    try:
        from captum.attr import IntegratedGradients
        
        model = get_model()
        ig = IntegratedGradients(model)
        
        x = input_tensor.clone()
        
        if target is None:
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=1).item()
        elif isinstance(target, str):
            target = int(target)
        
        # Create baseline (black image for MNIST)
        baseline = torch.zeros_like(x)
        
        attributions = ig.attribute(x, baselines=baseline, target=target, n_steps=n_steps)
        
        attr = torch.abs(attributions[0, 0]).cpu().numpy()
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
            
        return attr
        
    except ImportError:
        print("Captum not installed. Falling back to manual implementation")
        return integrated_gradients(input_tensor, target, n_steps)
    except Exception as e:
        print(f"Error with Integrated Gradients Captum: {e}")
        return integrated_gradients(input_tensor, target, n_steps)

def blur_integrated_gradients_captum(input_tensor, target=None, n_steps=50, max_blur=10):
    """Blur Integrated Gradients using Captum"""
    try:
        from captum.attr import IntegratedGradients
        from PIL import Image, ImageFilter
        
        model = get_model()
        ig = IntegratedGradients(model)
        
        x = input_tensor.clone()
        
        if target is None:
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=1).item()
        elif isinstance(target, str):
            target = int(target)
        
        # Create blurred baseline
        img_pil = Image.fromarray((x[0, 0].cpu().numpy() * 255).astype(np.uint8), mode='L')
        blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=max_blur))
        blurred_tensor = torch.tensor(np.array(blurred_pil) / 255.0, dtype=torch.float32)
        baseline = blurred_tensor.unsqueeze(0).unsqueeze(0)
        
        attributions = ig.attribute(x, baselines=baseline, target=target, n_steps=n_steps)
        
        attr = torch.abs(attributions[0, 0]).cpu().numpy()
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
            
        return attr
        
    except ImportError:
        print("Captum not installed. Using regular integrated gradients")
        return integrated_gradients_captum(input_tensor, target, n_steps)
    except Exception as e:
        print(f"Error with Blur IG Captum: {e}")
        return integrated_gradients_captum(input_tensor, target, n_steps)