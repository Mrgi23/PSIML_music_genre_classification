import torch
import os 
from preprocessing import Preprocessing
from torch.utils.data import DataLoader
from torchaudio.datasets import GTZAN
from dataset import AudioDataset
from torch.nn import Module
from torch.utils.data import random_split
from torch.nn.functional import sigmoid, softmax

class MusicModel(Module):
    def __init__(self):
        super(MusicModel, self).__init__()
        self.conv = torch.nn.Conv2d(3,20)
        self.layer = torch.nn.Linear(in_features=20, out_features=10, bias=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        x = softmax(x)
        return x





        