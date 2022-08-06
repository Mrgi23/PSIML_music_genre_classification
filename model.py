from turtle import forward
import torch
from torch.nn import Module, Conv2d, Linear, Flatten, MaxPool2d, AvgPool2d, Dropout, AdaptiveAvgPool2d, LazyLinear, BatchNorm2d,LazyBatchNorm1d
from torch.nn.functional import softmax, relu
import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
class SpectogramModule(Module):
    def __init__(self):
        super(SpectogramModule, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(513,4), groups=1)
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=(1,4), groups=128)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1,4), groups=256)
        self.linear1 = Linear(in_features=256, out_features=300)
        self.linear2 = Linear(in_features=300, out_features=150)
        self.batchnorm1 = BatchNorm2d(num_features=128)
        self.batchnorm2 = BatchNorm2d(num_features=256)
        self.batchnorm3 = BatchNorm2d(num_features=256)
        self.avgpolling = AvgPool2d(kernel_size=(1,26))
        self.maxpolling = MaxPool2d(kernel_size=(1,2))
        self.maxpolling_last = MaxPool2d(kernel_size=(1,26))
        self.flat = Flatten()
        
        self.dropout = Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)

        x = relu(x)
        x_1 = self.maxpolling_last(x)
        x_2 = self.avgpolling(x)
        x = self.flat(x_1 + x_2)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x
class MFCCModuel(Module):
    def __init__(self):
        super(MFCCModuel, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), groups=1)
        self.conv2 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), groups=1)
        self.avgpolling = AvgPool2d(kernel_size=(3,3),count_include_pad=True)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), groups=1)
        self.conv4 = Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), groups=1)
        self.globalpolling = AdaptiveAvgPool2d((1,1))
        self.flat = Flatten()
        self.dropout = Dropout(0.2)
        self.batchnorm1 = BatchNorm2d(num_features=256)
        self.batchnorm2 = BatchNorm2d(num_features=256)
        self.batchnorm3 = BatchNorm2d(num_features=256)
        self.batchnorm4 = BatchNorm2d(num_features=512)
        self.linear1 = Linear(in_features=512, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=128)
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = relu(x)
        x = self.avgpolling(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = relu(x)
        x = self.globalpolling(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x
        
class MusicModel(Module):
    def __init__(self,data_used):
        super(MusicModel, self).__init__()
        self.spectogram = None
        self.mfcc = None
        if 'spectrogram' in data_used:
            self.spectogram = SpectogramModule()
            data_used.remove('spectrogram')
        if 'mfcc' in data_used:
            self.mfcc = MFCCModuel()
            data_used.remove('mfcc')
        self.classifier = LazyLinear(out_features=10)
        self.scaler = data_used
        self.batchnorm = LazyBatchNorm1d()
    def forward(self, x):
        features = []
        if self.spectogram is not None:
            features.append(self.spectogram(x['spectrogram']))
        if self.mfcc is not None:
            features.append(self.mfcc(x['mfcc']))
        for s in self.scaler:
            features.append(x[s])
        x = torch.cat(tuple(features),1)
        x = self.batchnorm(x)
        x = relu(x)
        x = self.classifier(x)
        x = softmax(x,1)
        return x 

    def fit(self, args):

        num_epochs = args['num_epochs']
        train_dataloader = args['train_dataloader']
        validation_dataloader = args['validation_dataloader']
        device = args['device']
        optimizer = args['optimizer']
        loss_func = args['loss_func']

        best_loss = 1000
        best_model = self.state_dict()
        best_model_cnt = 0
        break_flag = False
        
        for epoch in range(num_epochs):
            self.train()
            for i, (data, label) in enumerate(train_dataloader):
                if isinstance(data,dict):
                    for key in data:
                        data[key] = data[key].to(device)
                else:
                    data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()

            
            correct = 0
            total = 0
            total_loss = 0
            self.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(validation_dataloader):
                    if isinstance(data,dict):
                        for key in data:
                            data[key] = data[key].to(device)
                    else:
                        data = data.to(device)
                    label = label.to(device)

                    output = self(data)
                    total_loss += loss_func(output, label)
                    _, predicted = torch.max(output, 1)
                    _, label = torch.max(label, 1)

                    total += label.size(0)
                    correct += (predicted == label).sum()
                
                validation_accuracy = 100*float(correct)/total
                if total_loss <  best_loss:
                    best_model = copy.deepcopy(self.state_dict())
                    best_model_cnt = 0
                    best_loss = total_loss
                else:
                    best_model_cnt += 1
                    if best_model_cnt == 5:
                        break_flag = True
                        break
                print(f'epoch: {epoch}\t   validation accuracy: {validation_accuracy:.4f}%\t loss: {total_loss:.4f}' )
            if break_flag:
                break
        self.load_state_dict(best_model)
    
    def predict(self, args):
        correct = 0
        total = 0

        test_dataloader = args['test_dataloader']
        device = args['device']

        self.eval()
        with torch.no_grad():
            for i, (data, label) in enumerate(test_dataloader):
                if isinstance(data,dict):
                    for key in data:
                        data[key] = data[key].to(device)
                else:
                    data = data.to(device)
                label = label.to(device)

                output = self(data)
                _, predicted = torch.max(output, 1)
                _, label = torch.max(label, 1)

                total += label.size(0)
                correct += (predicted == label).sum()
            classes = args['classes']
            classes_genre = args['classes_genre']
            label = [classes_genre[int(l)] for l in label]
            predicted = [classes_genre[int(p)] for p in predicted]
            cm = confusion_matrix(label, predicted, normalize='true')
            fig, ax = plt.subplots()
            fig.set_size_inches(9,9)
            ConfusionMatrixDisplay(cm, display_labels=list(classes)).plot(ax=ax)
            plt.savefig( f"confusion_matrix_seed_{args['seed']}.png")
            self.test_accuracy = 100*float(correct)/total
            print(f'testing accuracy: {self.test_accuracy}%' )
