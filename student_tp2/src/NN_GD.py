import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
import torch.nn.functional as F

# writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1,1)


class DummyNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.MSELoss()
        self.layers = torch.nn.Sequential(self.linear1, self.activation, self.linear2)
    
    def forward(self, x):
        y = self.layers(x)
        return self.layers(x)
