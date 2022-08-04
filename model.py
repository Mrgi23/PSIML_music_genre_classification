from torch.nn import Module,Conv2d,LazyLinear,Flatten,MaxPool2d,AvgPool2d
from torch.nn.functional import sigmoid,softmax,relu
class MusicModel(Module):
    def __init__(self,input_shape):
        super(MusicModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=128, kernel_size=(513,4), groups=1)
        self.maxpolling = MaxPool2d(kernel_size=(1,2))
        self.conv2 = Conv2d(in_channels=128, out_channels=256, kernel_size=(1,4), groups=128)
        self.conv3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(1,4), groups=256)
        self.maxpolling_last = MaxPool2d(kernel_size=(1,26))
        self.avgpolling = AvgPool2d(kernel_size=(1,26))
        self.flat = Flatten()
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
        x = self.flat(x_1 + x_2)
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        x = relu(x)
        x = self.linear3(x)
        x = relu(x)

        x = softmax(x)
        return x





        