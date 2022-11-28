from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32


PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries temporelles
model = RNN(in_dim=DIM_INPUT, h_dim=16, out_dim=DIM_INPUT)
optim = torch.optim.Adam(model.parameters())

for epoch in range(100):
    epoch_loss = 0
    n_samples = 0
    acc = 0
    for _, (x, y) in enumerate(data_train):
        optim.zero_grad()
        init_hstate = torch.zeros(x.size(0)*x.size(2), 16)
        states = model(x.transpose(0, 1).flatten(1, 2), init_hstate)

        yhat = model.decode(states)

        loss = torch.nn.functional.mse(yhat, y.transpose(0, 1).flatten(1, 2))
        
        # Log
        epoch_loss += loss.item()
        n_samples += x.size(0)
        acc += (torch.argmax(x) == torch.argmax(yhat)).sum()
        
        loss.backward()
        optim.step()
    if epoch % EVAL_EVERY == 0:
        epoch_loss_val = 0
        n_samples_val = 0
        acc_val = 0
        for _, (x, y) in enumerate(data_test):
            init_hstate = torch.zeros(x.size(0), 16)
            states = model(x.transpose(0, 1)[:-HORIZON], init_hstate)

            yhat = model.decode(states[-1])
            loss = torch.nn.functional.cross_entropy(yhat, y)
            
            # Log
            epoch_loss_val += loss.item()
            n_samples_val += x.size(0)
            acc_val += (torch.argmax(y) == torch.argmax(yhat)).sum()
        print(f"Epoch: {epoch}, Test Loss: {epoch_loss_val/n_samples_val}")
    print(f"Epoch: {epoch}, Train Loss: {epoch_loss/n_samples}")