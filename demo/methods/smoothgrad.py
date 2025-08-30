import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np

from model import get_model
from .vanilla_gradient import vanilla_gradient

def smoothgrad(input_tensor, target=None, n_samples=50, noise_level=0.1):
    """Manual SmoothGrad implementation"""
    model = get_model()
    
    if target is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    gradients = []
    
    for _ in range(n_samples):
        # Add noise to input
        noise = torch.randn_like(input_tensor) * noise_level
        noisy_input = input_tensor + noise
        noisy_input = torch.clamp(noisy_input, 0, 1)  # Keep in valid range
        
        # Get gradient for noisy input
        x = noisy_input.clone().requires_grad_(True)
        logits = model(x)
        score = logits[0, target]
        score.backward()
        
        grad = x.grad[0, 0].abs().cpu().numpy()
        gradients.append(grad)
    
    # Average all gradients
    avg_grad = np.mean(gradients, axis=0)
    
    # Normalize
    avg_grad = avg_grad - avg_grad.min()
    if avg_grad.max() > 0:
        avg_grad = avg_grad / avg_grad.max()
    
    return avg_grad

def smoothgrad_captum(input_tensor, target=None, n_samples=50, noise_level=0.1):
    """SmoothGrad using Captum"""
    try:
        from captum.attr import NoiseTunnel, Saliency
        
        model = get_model()
        saliency = Saliency(model)
        noise_tunnel = NoiseTunnel(saliency)
        
        x = input_tensor.clone()
        
        if target is None:
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=1).item()
        elif isinstance(target, str):
            target = int(target)
        
        attributions = noise_tunnel.attribute(
            x, 
            target=target, 
            n_samples=n_samples,
            stdevs=noise_level,
            abs=True
        )
        
        attr = attributions[0, 0].cpu().numpy()
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
            
        return attr
        
    except ImportError:
        print("Captum not installed. Falling back to manual implementation")
        return smoothgrad(input_tensor, target, n_samples, noise_level)
    except Exception as e:
        print(f"Error with SmoothGrad Captum: {e}")
        return smoothgrad(input_tensor, target, n_samples, noise_level)