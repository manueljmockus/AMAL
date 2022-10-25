from utils import RNN, device,SampleMetroDataset
import torch
from torch import nn
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
#Taille du batch
BATCH_SIZE = 32

LATENT = 64

PATH = "/home/manuel/AMAL/student_tp4/data/"

#####################
"""
ideas:
init h0 =
    vecteur de 0 ou de 1 (attention avec init a 0 car on introduit un biais)
    normal(0,1)
    init avec x0
    
Pourquoi on untilse pas RELU :
    relu n'est pas borné (on multiplie T fois notre reseau si val  =2 ca explose)
    mieux tanh ou sigmoide (bornee )
"""


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)

rnn = RNN(LATENT, 1 ,CLASSES)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(rnn.parameters(), lr = 0.001)
NB_ITERATIONS = 100

for epoch in range(NB_ITERATIONS): 
    for x,y in data_train:
        optim.zero_grad()
        hidden = rnn(x)
        #y = y.type(torch.LongTensor)
        output = rnn.decode(hidden[:,-1, :])
        loss = criterion(output, y)
        p = torch.max(output, axis= 1)[1]
        acc = torch.sum((torch.max(output, axis= 1)[1] == y).int())/BATCH_SIZE

        loss.backward()
        optim.step()
        print(f"Itérations {epoch}: loss {loss}, acc {acc}")
    

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence




