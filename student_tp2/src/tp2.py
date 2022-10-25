import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm

lr = 1e-9
data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float, requires_grad= True)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.size(0)

def model(X,W,b):
    return X @ W.t() + b
writer = SummaryWriter()


def GD(datax, datay, epochs, lr):
    W = torch.rand(datay.size(1), datax.size(1), dtype=torch.float ,requires_grad= True)
    b = torch.rand(datax.size(1), dtype=torch.float ,requires_grad= True)

    for n_iter in range(epochs):
        
        preds = model(datax,W,b)
        loss = mse(preds, datay)
        
        loss.backward()
        writer.add_scalar('Loss/train', loss, n_iter)
        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")
        with torch.no_grad():
            W -= W.grad * lr
            b -= b.grad * lr
            W.grad.data.zero_()
            b.grad.data.zero_()
    return


def mini_batchGD(datax, datay, epochs, lr):


    W = torch.rand(datay.size(1), datax.size(1), dtype=torch.float ,requires_grad= True)
    b = torch.rand(datax.size(1), dtype=torch.float ,requires_grad= True)

    for n_iter in range(epochs):
        
        preds = model(datax,W,b)
        loss = mse(preds, datay)
        
        loss.backward()
        writer.add_scalar('Loss/train', loss, n_iter)
        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")
        with torch.no_grad():
            W -= W.grad * lr
            b -= b.grad * lr
            W.grad.data.zero_()
            b.grad.data.zero_()
    return
