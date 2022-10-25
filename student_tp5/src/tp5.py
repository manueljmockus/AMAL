
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    binary_mask = (target == padcar).type(torch.int)
    loss = cross_entropy(output, target,reduction='none')
    p = loss * binary_mask
    L = torch.sum(loss * binary_mask)
    return L
    

class RNN(nn.Module):
    def __init__(self, latent, dim, output):
        super().__init__()
        
        self.latent = latent
        
        self.linear_x = nn.Linear(dim, latent)
        self.linear_h = nn.Linear(latent, latent)
        self.linear_y = nn.Linear(latent, output)
        
        self.TanH = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
    def one_step(self, x, h: torch.Tensor):
        return self.TanH(self.linear_x(x) + self.linear_h(h))
        
    def forward(self, x, h = None):
        sequence_length = x.size(1)
        self.batch = x.size(0)
        
        if h is None:
            h = torch.normal(torch.zeros(self.batch, self.latent), torch.ones(self.batch, self.latent))
        # hidden = torch.zeros(self.batch, sequence_length, self.latent)
        
        for i in range(sequence_length):
            h = self.one_step(x[:, i, :], h)
            #hidden[:,i,:] = h
        
        output = self.decode(h)
        return output
    
    def decode(self, h):
        return self.Sigmoid(self.linear_y(h))


class LSTM(RNN):
    def __init__(self, latent, dim, output):
        super().__init__()
        
        self.latent = latent
        self.linear_x = nn.Linear(dim, latent)
        self.linear_h = nn.Linear(latent, latent)
        self.linear_y = nn.Linear(latent, output)
        
        self.TanH = nn.Tanh()
        self.Sigmoid = nn.Sigmoid()
        
    def one_step(self, x, h: torch.Tensor):
        x_h = torch.cat(x,h)
        ft = self.Sigmoid(self.linear_f(x_h))
        it = self.Sigmoid(self.linear_i(x_h))
        Ct = ft * (self.Ct_old + it) * self.TanH(self.linear_C(x_h))
        ot = self.Sigmoid(self.linear_o(x_h))
        ht = ot * self.TanH(Ct)
        return ht, ot

class GRU(nn.Module):
    #  TODO:  Implémenter un GRU



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
