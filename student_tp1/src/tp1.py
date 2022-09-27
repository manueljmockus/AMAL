# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

from tracemalloc import get_traced_memory
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        fwd = torch.linalg.norm(yhat - y)**2 / y.size(0)
        return fwd

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        grad_yhat = (2 / y.size(0)) * grad_output * (yhat - y)
        grad_y = (-2 / y.size(0)) * grad_output * (yhat - y)
        return grad_yhat, grad_y
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)


#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    @staticmethod
    def forward(ctx, X, W, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(X, W, b)
        return torch.matmul(X, W) + b

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        ## grad_output v
        X, W, b = ctx.saved_tensors
        grad_x = torch.matmul(grad_output, torch.transpose(W, 0, 1))
        grad_w = torch.matmul(torch.transpose(X, 0, 1), grad_output)
        grad_b = grad_output 
        return grad_x, grad_w, grad_b

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

