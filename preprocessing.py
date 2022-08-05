import torch
from torchaudio.datasets import GTZAN
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from torch.nn.functional import one_hot
from dataset import AudioDataset
import pickle
import numpy as np
class Preprocessing():
    
    def __init__(self):
        self.__dataset = None
        self.__transform = None
        self.__data = None
        self.__spectrograms = None
    
    def __load_file(self, ROOT, FOLDER):
        
        dataset = GTZAN(root=ROOT, folder_in_archive=FOLDER)
        classes = np.unique([name.split('.')[0]for name in  dataset._walker])
        self.classes = {c: torch.tensor(i) for i,c in enumerate(classes)}
        self.__dataset = dataset
        self.__data = []
    
    def __load_transform(self, NFFT):
        self.__transform_spec = Spectrogram(n_fft=NFFT, hop_length=NFFT//2, center=False, pad=24, power=1.0)
        self.__transform_amp2db = AmplitudeToDB()
        self.__spectrograms = []
    
    def __split(self, SIZE, SPLIT_PARTITIONS):
        for i in range(len(self.__dataset)):
            genre = self.__dataset[i][2]
            genre_onehot = one_hot(self.classes[genre], num_classes= len(self.classes)).type(torch.float32).to('cuda')
            signal = self.__dataset[i][0][:, :SIZE]
            dim = signal.shape[1]//SPLIT_PARTITIONS

            for j in range(SPLIT_PARTITIONS):
                tmp = signal[:, j*dim:(j+1)*dim]
                self.__data.append((tmp, genre_onehot))

                tmp = self.__transform_spec(tmp)
                tmp = self.__transform_amp2db(tmp)
                self.__spectrograms.append((tmp, genre_onehot))
        
    
    def preprocessing(self, args):
        try:
            with open('./pickle_data', 'rb') as pickle_in:
                dic = pickle.load(pickle_in)
                pickle_in.close()
            return None, dic['spectrograms']
        except FileNotFoundError:
            self.__load_file(ROOT=args['ROOT'], FOLDER=args['FOLDER'])
            self.__load_transform(NFFT=args['NFFT'])
            self.__split(SIZE=args['SIZE'], SPLIT_PARTITIONS=args['SPLIT_PARTITIONS'])

            audio_data = AudioDataset(data=self.__data)
            spectrograms = AudioDataset(data=self.__spectrograms)
            dic = {'spectrograms': spectrograms}
            with open('./pickle_data', 'wb') as pickle_out:
                pickle.dump(dic, pickle_out)
                pickle_out.close()
            return audio_data, spectrograms