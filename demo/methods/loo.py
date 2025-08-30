import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np

from model import get_model

def leave_one_out(input_tensor, target=None):
    """Leave-One-Out attribution"""
    model = get_model()
    
    if target is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    # Get original prediction
    with torch.no_grad():
        original_logits = model(input_tensor)
        original_prob = F.softmax(original_logits, dim=1)[0, target].item()
    
    # Create saliency map
    height, width = 28, 28
    saliency = np.zeros((height, width))
    
    # For each pixel, set it to 0 and measure change
    for i in range(height):
        for j in range(width):
            # Create modified input
            modified_input = input_tensor.clone()
            modified_input[0, 0, i, j] = 0
            
            # Get new prediction
            with torch.no_grad():
                modified_logits = model(modified_input)
                modified_prob = F.softmax(modified_logits, dim=1)[0, target].item()
            
            # Saliency is the drop in probability
            saliency[i, j] = original_prob - modified_prob
    
    # Normalize
    saliency = np.maximum(saliency, 0)  # Only positive contributions
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    
    return saliency

def leave_one_out_captum(input_tensor, target=None):
    """Leave-One-Out using Captum (Feature Ablation)"""
    try:
        from captum.attr import FeatureAblation
        
        model = get_model()
        ablation = FeatureAblation(model)
        
        x = input_tensor.clone()
        
        if target is None:
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=1).item()
        elif isinstance(target, str):
            target = int(target)
        
        # Create feature mask (each pixel is a feature)
        feature_mask = torch.arange(28*28).reshape(1, 1, 28, 28)
        
        attributions = ablation.attribute(x, target=target, feature_mask=feature_mask)
        
        attr = attributions[0, 0].cpu().numpy()
        attr = np.maximum(attr, 0)  # Only positive contributions
        if attr.max() > 0:
            attr = attr / attr.max()
            
        return attr
        
    except ImportError:
        print("Captum not installed. Falling back to manual implementation")
        return leave_one_out(input_tensor, target)
    except Exception as e:
        print(f"Error with Leave-One-Out Captum: {e}")
        return leave_one_out(input_tensor, target)