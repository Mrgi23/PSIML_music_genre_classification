from torch.utils.data import Dataset
class AudioDataset(Dataset):
    def __init__(self, data, transform, num_frames=660000 ) -> None:

        super().__init__()
        self.data = data
        self.transform = transform
        self.num_frames = num_frames
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        x,_,y =self.data[index]
        return (self.transform(x)[:,:self.num_frames],y)