from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MFCC
import pickle

class AudioDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.__data = data
        self.__transform_spec = None
        self.__transform_A2dB = None

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]
    
    def __load_transform(self, NFFT):
        self.__transform_spec = Spectrogram(n_fft=NFFT, hop_length=NFFT//2, center=False, pad=24, power=1.0)
        self.__transform_A2dB = AmplitudeToDB()
        # self.__transform_MFCC = MFCC(sample_rate=22500,n_mfcc=40,melkwargs={'n_fft':NFFT,'hop_length':NFFT//2,'center':False,'pad':28})
    
    def __split(self, SPLIT_PARTITIONS):
        data = []
        dim = self.__data[0][0].shape[1]//SPLIT_PARTITIONS
        for i in range(len(self.__data)):
            for j in range(SPLIT_PARTITIONS):
                tmp = self.__data[i][0][:, j*dim:(j+1)*dim]
                tmp = self.__transform_spec(tmp)
                tmp = self.__transform_A2dB(tmp)
                data.append((tmp, self.__data[i][1]))
        self.__data = data

    def preprocess(self, args):
        self.__load_transform(NFFT=args['NFFT'])
        self.__split(SPLIT_PARTITIONS=args['SPLIT_PARTITIONS'])