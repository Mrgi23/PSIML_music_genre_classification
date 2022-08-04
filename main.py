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
train_dataset,validation_dataset, test_dataset = random_split(dataset,lengths=[train_lenght,validation_lenght,test_lenght])
train_dataset[0]
train_dataloader = DataLoader(train_dataset,batch_size=32, shuffle=True)
x=next(iter(train_dataloader))[0]


model = MusicModel(x.shape)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())
num_of_epochs = 100
for epoch in range(num_of_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
print('done')
