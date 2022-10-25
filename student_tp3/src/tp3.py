from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

from pathlib import Path
from dataset import *
from tqdm.notebook import tqdm

# Téléchargement des données

from datamaestro import prepare_dataset

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/manuel/AMAL/student_tp2/src')

from NN_GD import *

BATCH_SIZE = 64
NB_ITERATIONS = 100

ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath  = Path("model3.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_train = DataLoader ( MyDataset(train_images,train_labels, normalize_images) , shuffle=True , batch_size=BATCH_SIZE)
criterion = nn.MSELoss()

if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
else:
    
    model = NN(train_images.shape[1], 32, 1)
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters())
    state = State(model, optim)
    
for epoch in tqdm(range(state.epoch, NB_ITERATIONS)):
    for x,y in data_train:
        state.optim.zero_grad()
        x = x.to(device)
        y = y.to(device)
        yhat = state.model(x)
        loss = criterion(yhat, y)
        
        loss.backward()
        state.optim.step()
        state.iteration += 1
        #print("loss", loss )
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)
