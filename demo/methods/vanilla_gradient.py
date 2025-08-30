import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np

from model import get_model 


def vanilla_gradient(input_tensor, target=None):
    """Vanilla gradient implementation using cached model"""
    model = get_model()
    
    x = input_tensor.clone().requires_grad_(True)
    logits = model(x)
    
    if target is None:
        target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    score = logits[0, target]
    score.backward()
    
    grad = x.grad[0, 0].abs().cpu().numpy()
    grad = grad - grad.min()
    if grad.max() > 0:
        grad = grad / grad.max()
    
    return grad

def vanilla_gradient_captum(input_tensor, target=None):
    """Vanilla gradient implementation using Captum library"""
    from captum.attr import Saliency
    
    model = get_model()
    
    # Captum's Saliency class implements vanilla gradients
    saliency = Saliency(model)
    
    # Prepare input - Captum handles device placement
    x = input_tensor.clone().requires_grad_(True)
    
    # Get target class if not provided
    if target is None:
        with torch.no_grad():
            logits = model(x)
            target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    # Get attributions (gradients)
    attributions = saliency.attribute(x, target=target, abs=True)
    
    # Convert to numpy and normalize
    attr = attributions[0, 0].cpu().numpy()
    attr = attr - attr.min()
    if attr.max() > 0:
        attr = attr / attr.max()
        
    return attr