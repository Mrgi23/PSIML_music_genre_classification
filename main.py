import torch
import os 
from preprocessing import Preprocessing
import torchaudio
from torchaudio.datasets import GTZAN
from torchaudio.transforms import MelSpectrogram,MFCC,Spectrogram
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import AudioDataset
from model import MusicModel

preproc = Preprocessing()
args = {'ROOT': './Data',
        'FOLDER': 'genres_original',
        'NFFT': 1024,
        'SIZE': 660000,
        'SPLIT_PARTITIONS': 10}
audio, dataset = preproc.preprocessing(args=args)
print('Preprocessing done!\n')

train_lenght = int(len(dataset)*0.6)
validation_lenght = int(len(dataset)*0.2)
test_lenght = len(dataset) - train_lenght - validation_lenght
train_dataset,validation_dataset, test_dataset = random_split(dataset,lengths=[train_lenght,validation_lenght,test_lenght])
print('Splitting done!\n')
train_dataset[0]
train_dataloader = DataLoader(train_dataset,batch_size=32, shuffle=True)
x=next(iter(train_dataloader))[0]
print('Loading done!\n')
model = MusicModel(x.shape)
a = model(x)
print('Forward done!\n')
print(a.shape)
