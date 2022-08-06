from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, AmplitudeToDB, MFCC
from torch import tensor
from librosa import feature as ft
import numpy as np
import pickle
class AudioDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.__data = data
        self.__transform_spec = None
        self.__transform_A2dB = None
        self.__transform_MFCC = None

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]
    
    def __load_transform(self, NFFT):
        self.__transform_spec = Spectrogram(n_fft=NFFT, hop_length=NFFT//2, center=False, pad=24, power=1.0)
        self.__transform_A2dB = AmplitudeToDB()
        self.__transform_MFCC = MFCC(sample_rate=22500,n_mfcc=13,melkwargs={'n_fft':2*NFFT,'hop_length':NFFT//2,'center':False,'pad':28})
    
    def __split(self, SPLIT_PARTITIONS, NFFT):
        data = []
        dim = self.__data[0][0].shape[1]//SPLIT_PARTITIONS
        for i in range(len(self.__data)):
            for j in range(SPLIT_PARTITIONS):
                data_dict = {}
                
                # Spectrogram
                data_sample = self.__data[i][0][:, j*dim:(j+1)*dim]
                tmp1 = self.__transform_spec(data_sample)
                tmp1 = self.__transform_A2dB(tmp1)
                data_dict['spectrogram'] = tmp1

                # MFFC
                tmp2 = self.__transform_MFCC(data_sample)
                data_dict['mfcc'] = tmp2.reshape(1,tmp2.shape[1]*2,-1)

                # Spectral centroid
                tmp3 = ft.spectral_centroid(y=data_sample.numpy()[0], n_fft=NFFT)
                mv1 = tensor([np.mean(tmp3)]).float()
                std1 = tensor([np.std(tmp3)]).float()
                data_dict['c_mv'] = mv1
                data_dict['c_std'] = std1

                # Spectral roll-off
                tmp4 = ft.spectral_rolloff(y=data_sample.numpy()[0], n_fft=NFFT)
                mv2 = tensor([np.mean(tmp4)]).float()
                std2 = tensor([np.std(tmp4)]).float()
                data_dict['r_mv'] = mv2
                data_dict['r_std'] = std2

                data.append((data_dict, self.__data[i][1]))
            print('{}.---------------------------------------------------------------------'.format(i))
        self.__data = data

    def preprocess(self, args):
        try:
            with open('./pickle_transform', 'rb') as pickle_in:
                dic = pickle.load(pickle_in)
                pickle_in.close()
            self.__data = dic['transform']
        except FileNotFoundError:
            self.__load_transform(NFFT=args['NFFT'])
            self.__split(SPLIT_PARTITIONS=args['SPLIT_PARTITIONS'], NFFT=args['NFFT'])
            dic = {'transform': self.__data}
            with open('./pickle_transform', 'wb') as pickle_out:
                pickle.dump(dic, pickle_out)
                pickle_out.close()