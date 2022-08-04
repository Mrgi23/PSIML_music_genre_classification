from torch.utils.data import Dataset

class AudioDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]