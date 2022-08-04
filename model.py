import torch
from torch.nn import Module,Conv2d,LazyLinear,Flatten,MaxPool2d,AvgPool2d
from torch.nn.functional import sigmoid,softmax,relu
from torch import flatten
class MusicModel(Module):

    def __init__(self):
        super(MusicModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(513,4), groups=1)
        self.maxpolling = MaxPool2d(kernel_size=(1,2))
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=(1,4), groups=128)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1,4), groups=256)
        self.maxpolling_last = MaxPool2d(kernel_size=(1,26))
        self.avgpolling = AvgPool2d(kernel_size=(1,26))
        self.linear1 = LazyLinear(out_features=300)
        self.linear2 = LazyLinear(out_features=150)
        self.linear3 = LazyLinear(out_features=10)

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
        x = flatten(x_1 + x_2, start_dim=1)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = relu(x)

        x = softmax(x, 1)
        return x

    def fit(self, args):
        
        for epoch in range(args['num_epochs']):
            self.train()
            for i, (data, label) in enumerate(args['train_dataloader']):
                data = data.to(args['device'])
                label = label[0].to(args['device'])

                args['optimizer'].zero_grad()
                output = self(data)
                loss = args['loss_func'](output, label)
                loss.backward()
                args['optimizer'].step()

                if (i+1) % (len(data)//args['batch_size']) == 0:
                    print(f'epoch: {epoch} - iter: {i+1} - batch_loss: {loss}')
            
            correct = 0
            total = 0

            self.eval()
            with torch.no_grad():
                for i, (data, label) in enumerate(args['validation_dataloader']):
                    data = data.to(args['device'])
                    label = label[0].to(args['device'])

                    output = self(data)
                    label_pred = softmax(output, 1)
                    _, predicted = torch.max(label_pred, 1)

                    total += label.size(0)
                    correct += (predicted == label).sum()
                
                validation_accuracy = 100*float(correct)/total
                print(f'epoch: {epoch}   validation accuracy: {validation_accuracy}%' )
