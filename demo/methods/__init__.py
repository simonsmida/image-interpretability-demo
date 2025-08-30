# File: demo/methods/__init__.py

from .vanilla_gradient import vanilla_gradient, null_saliency, vanilla_gradient_captum
#from .smoothgrad import smoothgrad
#from .integrated_gradients import integrated_gradients
#from .grad_cam import grad_cam
#from .loo import leave_one_out
#from .blur_ig import blur_integrated_gradients

METHODS = {
    "Null (placeholder)": null_saliency,
    "Vanilla Gradient": vanilla_gradient,
    "Vanilla Gradient (Captum)": vanilla_gradient_captum,
    # "SmoothGrad": smoothgrad,
    # "Integrated Gradients": integrated_gradients,
    # "Grad-CAM": grad_cam,
    # "Leave-One-Out": leave_one_out,
    # "Blur IG": blur_integrated_gradients,
}

__all__ = ["METHODS"]
