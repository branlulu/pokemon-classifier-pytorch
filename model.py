import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import activation
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d

# Helper function to print dimensions of layer
class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x
        
# Return model architecture
def get_model():
    model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.3),
            nn.Flatten(),
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 150),
            nn.Softmax(dim=1)
        )

    return model


random_data = torch.rand((1, 3, 256, 256))