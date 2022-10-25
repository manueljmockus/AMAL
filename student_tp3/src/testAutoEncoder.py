from dataset import *

from datamaestro import prepare_dataset
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import datetime

ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

savepath  = Path("model3.pch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_train = DataLoader ( MyDataset(train_images,train_labels, normalize_images) , shuffle=True , batch_size=64)
data_test = DataLoader ( MyDataset(test_images,test_labels, normalize_images) , shuffle=True , batch_size=1)
criterion = nn.MSELoss()



AE = AutoEncoder(input_dims = 28, hidden_dims = 8)
criterion = nn.MSELoss()

AE = AE.to(device)
optim = torch.optim.Adam(AE.parameters())

for epoch in tqdm(range(20)):
    for x,y in data_train:
        optim.zero_grad()
        x = x.to(device)
        xhat = AE(x)
        loss = criterion(xhat, x)
        
        loss.backward()
        optim.step()
        #
    print("loss", loss)
    
i = 0
encoded_test = []
AE = AE.to(device)
for x,y in data_test:
    i+=1
    x = x.to(device)
    enc = AE(x)
    if i == 1:
        writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        after = torch.tensor(enc[0:10]).unsqueeze(1).repeat(1,3,1,1).double()
        before = torch.tensor(x[0:10]).unsqueeze(1).repeat(1,3,1,1).double()
        # Permet de fabriquer une grille d'images
        before = make_grid(before)
        after = make_grid(after)
        # Affichage avec tensorboard
        writer.add_image(f'before', before, 0)
        writer.add_image(f'after', after, 1)