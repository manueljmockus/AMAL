from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 32
# Interval between evaluation
EVAL_EVERY = 10

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

model = RNN(in_dim=DIM_INPUT, h_dim=16, out_dim=CLASSES)
optim = torch.optim.Adam(model.parameters())

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence
for epoch in range(10):
    epoch_loss = 0
    n_samples = 0
    acc = 0
    for _, (x, y) in enumerate(data_train):
        print(x.size(), y.size())
        assert False
        optim.zero_grad()
        init_hstate = torch.zeros(x.size(0), 16) 
        states = model(x.transpose(0, 1), init_hstate)

        yhat = model.decode(states[-1])

        loss = torch.nn.functional.cross_entropy(yhat, y)
        
        # Log
        epoch_loss += loss.item()
        n_samples += x.size(0)
        acc += (torch.argmax(y) == torch.argmax(yhat)).sum()
        
        loss.backward()
        optim.step()
    if epoch % EVAL_EVERY == 0:
        epoch_loss_val = 0
        n_samples_val = 0
        acc_val = 0
        for _, (x, y) in enumerate(data_test):
            init_hstate = torch.zeros(x.size(0), 16)
            states = model(x.transpose(0, 1), init_hstate)

            yhat = model.decode(states[-1])
            loss = torch.nn.functional.cross_entropy(yhat, y)
            
            # Log
            epoch_loss_val += loss.item()
            n_samples_val += x.size(0)
            acc_val += (torch.argmax(y) == torch.argmax(yhat)).sum()
        print(f"Epoch: {epoch}, Test Loss: {epoch_loss_val/n_samples_val}")
    print(f"Epoch: {epoch}, Train Loss: {epoch_loss/n_samples}")