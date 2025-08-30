import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
import cv2

from model import get_model

def grad_cam(input_tensor, target=None, target_layer_name="conv2"):
    """Manual Grad-CAM implementation"""
    model = get_model()
    
    if target is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target = logits.argmax(dim=1).item()
    elif isinstance(target, str):
        target = int(target)
    
    # Hook to capture gradients and activations
    gradients = None
    activations = None
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    # Register hooks on target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == target_layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Layer {target_layer_name} not found, using conv2")
        target_layer = model.conv2
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    input_tensor.requires_grad_(True)
    logits = model(input_tensor)
    score = logits[0, target]
    
    # Backward pass
    score.backward()
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Generate CAM
    if gradients is not None and activations is not None:
        # Global average pooling of gradients
        weights = torch.mean(gradients[0], dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().detach().numpy()
        
        return cam
    else:
        return np.zeros((28, 28), dtype=np.float32)

def grad_cam_captum(input_tensor, target=None):
    """Grad-CAM using Captum"""
    try:
        from captum.attr import GradCam
        
        model = get_model()
        
        # For small CNN, use the last conv layer
        gradcam = GradCam(model, model.conv2)
        
        x = input_tensor.clone()
        
        if target is None:
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=1).item()
        elif isinstance(target, str):
            target = int(target)
        
        attributions = gradcam.attribute(x, target=target)
        
        attr = attributions[0, 0].cpu().numpy()
        attr = attr - attr.min()
        if attr.max() > 0:
            attr = attr / attr.max()
            
        return attr
        
    except ImportError:
        print("Captum not installed. Falling back to manual implementation")
        return grad_cam(input_tensor, target)
    except Exception as e:
        print(f"Error with Grad-CAM Captum: {e}")
        return grad_cam(input_tensor, target)