import torch
import os 
from preprocessing import Preprocessing
import torchaudio
from torchaudio.datasets import GTZAN
from torchaudio.transforms import MelSpectrogram,MFCC,Spectrogram
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import AudioDataset
#from model import MusicModel
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
train_dataset[0]
train_dataloader = DataLoader(train_dataset,batch_size=32, shuffle=True)
x=next(iter(train_dataloader))[0]
model = MusicModel(x.shape)
model(x)
print('done')
