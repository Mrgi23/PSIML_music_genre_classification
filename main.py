from pickletools import optimize
import torch
import os 
from preprocessing import Preprocessing
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from dataset import AudioDataset
from model import MusicModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
preproc = Preprocessing()
args = {'ROOT': './Data',
        'FOLDER': 'genres_original',
        'NFFT': 1024,
        'SIZE': 660000,
        'SPLIT_PARTITIONS': 10}
audio,dataset = preproc.preprocessing(args=args)

train_lenght = int(len(dataset)*0.6)
validation_lenght = int(len(dataset)*0.2)
test_lenght = len(dataset) - train_lenght - validation_lenght
train_dataset, validation_dataset, test_dataset = random_split(dataset,lengths=[train_lenght,validation_lenght,test_lenght])
print('Splitting done!\n')

model = MusicModel()

num_epochs = 2
batch_size = 32
device = 'cpu'
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

loss_func = CrossEntropyLoss()
lr = 0.01
optimizer = Adam(model.parameters(), lr=lr)
args = {'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device,
        'train_dataloader': train_dataloader,
        'validation_dataloader': validation_dataloader,
        'loss_func': loss_func,
        'optimizer': optimizer}

print('Loading done!\n')
model.fit(args=args)
print('Training done!\n')
