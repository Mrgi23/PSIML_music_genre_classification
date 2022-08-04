import torch
import os 
from preprocessing import Preprocessing
import torchaudio
from torchaudio.datasets import GTZAN
from torchaudio.transforms import MelSpectrogram,MFCC,Spectrogram
from torch.utils.data import DataLoader
from dataset import AudioDataset
from torch.nn import Module
from torch.utils.data import random_split
from torch.nn.functional import sigmoid,softmax
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

path = os.path.join('Data','genres_original')
dataset = GTZAN('Data',folder_in_archive='genres_original')

sample_rate = 22050

n_fft = 1024
win_length = None
hop_length = 512

# define transformation
spectrogram = Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

dataset = AudioDataset(dataset,spectrogram)

train_lenght = int(len(dataset)*0.6)
validation_lenght = int(len(dataset)*0.2)
test_lenght = len(dataset) - train_lenght - validation_lenght
train_dataset,validation_dataset, test_dataset = random_split(dataset,lengths=[train_lenght,validation_lenght,test_lenght])
train_dataloader = DataLoader(train_dataset,batch_size=32, shuffle=True)
print('done')





        