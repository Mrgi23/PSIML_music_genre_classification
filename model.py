from curses.ascii import SP
from pickletools import optimize
import torch
from torch.nn import Module, Conv2d, Linear, Flatten, MaxPool2d, AvgPool2d, Dropout, AdaptiveAvgPool2d
from torch.nn.functional import softmax, relu
import copy
class SpectogramModule(Module):
    def __init__(self):
        super(SpectogramModule, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(513,4), groups=1)
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=(1,4), groups=128)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1,4), groups=256)
        self.linear1 = Linear(in_features=256, out_features=300)
        self.linear2 = Linear(in_features=300, out_features=150)
        self.linear3 = Linear(in_features=150, out_features=10)

        self.avgpolling = AvgPool2d(kernel_size=(1,26))
        self.maxpolling = MaxPool2d(kernel_size=(1,2))
        self.maxpolling_last = MaxPool2d(kernel_size=(1,26))
        self.flat = Flatten()
        
        self.dropout = Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = relu(x)
        x_1 = self.maxpolling_last(x)
        x_2 = self.avgpolling(x)
        x = self.flat(x_1 + x_2)
        x = self.linear1(x)
        x = self.dropout(x)
        x = relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = relu(x)
        x = self.linear3(x)
        x = self.dropout(x)
        x = relu(x)

        x = softmax(x, 1)
        return x
class MFCCModuel():
    def __init__(self):
        super(MFCCModuel, self).__init__()

        self.conv1 = Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), groups=1)
        self.conv2 = Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), groups=1)
        self.avgpolling = AvgPool2d(kernel_size=(3,3),count_include_pad=True)
        self.conv3 = Conv2d(in_channels=1, out_channels=256, kernel_size=(3,3), groups=1)
        self.conv4 = Conv2d(in_channels=1, out_channels=512, kernel_size=(4,4), groups=1)
        self.globalpolling = AdaptiveAvgPool2d((1,1))
        self.flat = Flatten()
        self.linear1 = Linear(in_features=512, out_features=256)
        self.linear2 = Linear(in_features=256, out_features=128)
        self.linear3 = Linear(in_features=128, out_features=10)
    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.avgpolling(x)
        x = self.conv3(x)
        x = relu(x)
        x = self.conv4(x)
        x = relu(x)
        x = self.globalpolling(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = softmax(x)
        
class MusicModel(Module):
    def __init__(self):
        super(MusicModel, self).__init__()
        self.spectogram = SpectogramModule()
        self.mfcc = MFCCModuel()

    def fit(self, args):

        num_epochs = args['num_epochs']
        train_dataloader = args['train_dataloader']
        validation_dataloader = args['validation_dataloader']
        device = args['device']
        optimizer = args['optimizer']
        loss_func = args['loss_func']

        best_accuracy = 0
        best_model = self.state_dict()
        best_model_cnt = 0
        break_flag = False
        
        for epoch in range(num_epochs):
            self.train()
            for i, (data, label) in enumerate(train_dataloader):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                output = self(data)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()

            
            correct = 0
            total = 0

            self.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(validation_dataloader):
                    data = data.to(device)
                    label = label.to(device)

                    output = self(data)
                    _, predicted = torch.max(output, 1)
                    _, label = torch.max(label, 1)

                    total += label.size(0)
                    correct += (predicted == label).sum()
                
                validation_accuracy = 100*float(correct)/total
                if validation_accuracy > best_accuracy:
                    best_accuracy = validation_accuracy
                    best_model = copy.deepcopy(self.state_dict())
                    best_model_cnt = 0
                else:
                    best_model_cnt += 1
                    if best_model_cnt == 10:
                        break_flag = True
                        break
                print(f'epoch: {epoch}   validation accuracy: {validation_accuracy}%' )
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
                data = data.to(device)
                label = label.to(device)

                output = self(data)
                _, predicted = torch.max(output, 1)
                _, label = torch.max(label, 1)

                total += label.size(0)
                correct += (predicted == label).sum()
            
            self.test_accuracy = 100*float(correct)/total
            print(f'testing accuracy: {self.test_accuracy}%' )
