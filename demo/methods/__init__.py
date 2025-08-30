# File: demo/methods/__init__.py

from .vanilla_gradient import vanilla_gradient, vanilla_gradient_captum
from .smoothgrad import smoothgrad, smoothgrad_captum
from .integrated_gradients import integrated_gradients, integrated_gradients_captum, blur_integrated_gradients_captum
from .grad_cam import grad_cam, grad_cam_captum
from .loo import leave_one_out, leave_one_out_captum

METHODS = {
    "Vanilla Gradient": vanilla_gradient,
    "Vanilla Gradient (Captum)": vanilla_gradient_captum,
    "SmoothGrad": smoothgrad,
    "SmoothGrad (Captum)": smoothgrad_captum,
    "Integrated Gradients": integrated_gradients,
    "Integrated Gradients (Captum)": integrated_gradients_captum,
    "Grad-CAM": grad_cam,
    "Grad-CAM (Captum)": grad_cam_captum,
    "Leave-One-Out": leave_one_out,
    "Leave-One-Out (Captum)": leave_one_out_captum,
    "Blur IG (Captum)": blur_integrated_gradients_captum,
}

__all__ = ["METHODS"]