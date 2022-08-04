from torch.nn import Module,Conv2d,Linear
from torch.nn.functional import sigmoid,softmax
class MusicModel(Module):
    def __init__(self):
        super(MusicModel, self).__init__()
        self.conv = Conv2d(3,20,3)
        self.layer = Linear(in_features=20, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        x = softmax(x)
        return x





        