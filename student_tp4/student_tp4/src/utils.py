import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn = nn.RNN(10, 20, 2)

class RNN(nn.Module):
    def __init__(self, latent, dim, output):
        super().__init__()
        
        self.latent = latent
        
        self.linear_x = nn.Linear(dim, latent)
        self.linear_h = nn.Linear(latent, latent)
        self.linear_y = nn.Linear(latent, output)
        
        self.TanH = nn.Tanh()
        self.Softmax = nn.Softmax()
        
    def one_step(self, x, h: torch.Tensor):
        """_summary_

        Args:
            x_batch (_type_): batch * dim
            h (_type_): latent

        Returns:
            _type_: batch × latent
        """
        return self.TanH(self.linear_x(x) + self.linear_h(h))
        
    def forward(self, x, h = None):
        sequence_length = x.size(1)
        self.batch = x.size(0)
        
        if h is None:
            h = torch.normal(torch.zeros(self.batch, self.latent), torch.ones(self.batch, self.latent))
        hidden = torch.zeros(self.batch, sequence_length, self.latent)
        
        for i in range(sequence_length):
            new_h = self.one_step(x[:, i, :], h)
            hidden[:,i,:] = new_h
            h = new_h
        
        return hidden
    
    def decode(self, h):
        return self.Softmax(self.linear_y(h))
        
        

class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

