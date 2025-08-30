import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
from pathlib import Path

# -------------------- configuration --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn.pth"

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)               # /2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)            # 14x14
        x = F.relu(self.conv2(x))
        x = self.pool(x)            # 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

_model = None

def load_model(path: Optional[str] = None):
    """Load model with smart path handling and caching"""
    global _model
    if _model is not None:
        return _model
    
    model = SmallCNN().to(DEVICE)
    
    # Use provided path or default MODEL_PATH
    model_path = path if path else MODEL_PATH
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Using untrained model")
    else:
        print(f"Model file not found at {model_path}, using untrained model")
    
    model.eval()
    _model = model
    return _model

def get_model():
    """Get the cached model instance"""
    return load_model()  # Will use default path and caching