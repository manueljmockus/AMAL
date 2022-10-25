from dataclasses import dataclass
from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

class MyDataset(Dataset):
    def __init__(self, data, labels, transform) -> None:
        super().__init__()
        self.data = transform(data)
        self.labels = labels.astype(np.float32)
    
    def __getitem__(self, index):
        return (self.data[index], self.labels[index])
    
    def __len__(self):
        return self.data.shape[0]

def normalize_images(images):
    images = images.astype(np.float32)/ 255.
    return np.array(images)
        
        

class State:
    def __init__(self, model, optim) -> None:
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0
        
class AutoEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        
        for i, dim in enumerate(kwargs["dims"]):
            layer = nn.Linear()
        encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_dims"], out_features=kwargs["hidden_dims"]
        )
        encoder_output_layer = nn.Linear(
            in_features=kwargs["hidden_dims"], out_features=kwargs["hidden_dims"]
        )
        decoder_hidden_layer = nn.Linear(
            in_features=kwargs["hidden_dims"], out_features=kwargs["hidden_dims"]
        )
        decoder_output_layer = nn.Linear(
            in_features=kwargs["hidden_dims"], out_features=kwargs["input_dims"]
        )
        self.encoder = nn.Sequential(encoder_hidden_layer,
                                     nn.ReLU(),
                                     encoder_output_layer,
                                     nn.ReLU(),
                                     )
        self.decoder = nn.Sequential(decoder_hidden_layer,
                                     nn.Sigmoid(),
                                     decoder_output_layer,
                                     nn.Sigmoid(),
                                     )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded