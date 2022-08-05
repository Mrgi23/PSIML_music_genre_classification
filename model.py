from pickletools import optimize
import torch
from torch.nn import Module, Conv2d, Linear, Flatten, MaxPool2d,AvgPool2d
from torch.nn.functional import softmax, relu
import copy
class MusicModel(Module):

    def __init__(self):
        super(MusicModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(513,4), groups=1)
        self.maxpolling = MaxPool2d(kernel_size=(1,2))
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=(1,4), groups=128)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1,4), groups=256)
        self.maxpolling_last = MaxPool2d(kernel_size=(1,26))
        self.avgpolling = AvgPool2d(kernel_size=(1,26))
        self.flat = Flatten()
        self.linear1 = Linear(in_features=256, out_features=300)
        self.linear2 = Linear(in_features=300, out_features=150)
        self.linear3 = Linear(in_features=150, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.maxpolling(x)
        x = self.conv3(x)
        x = relu(x)
        x_1 = self.maxpolling_last(x)
        x_2 = self.avgpolling(x)
        x = self.flat(x_1 + x_2)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = relu(x)

        x = softmax(x, 1)
        return x

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
