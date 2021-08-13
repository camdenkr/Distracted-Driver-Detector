import sys
from torchvision import models
import pandas as pd
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent)+"/..")
import matplotlib.pyplot as plt
import torch.nn as nn

''' 
Loads and returns models
-------
model_choice: name of a model to be loaded
'''
def get_model(model_choice):
    if(model_choice == "vgg16"):
        print("vgg16")
        model = models.vgg16(pretrained=True)
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
        
    if(model_choice == "alexNet"):
        print("alexNet")
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    if(model_choice == "googlenet"):
        print("googlenet")
        model = models.googlenet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10), nn.LogSoftmax(dim=1))
    return model