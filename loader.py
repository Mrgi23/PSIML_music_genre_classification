import torch
from torchaudio.datasets import GTZAN
from torch.nn.functional import one_hot
from dataset import AudioDataset
import pickle
import numpy as np

class Loader():    
    def __init__(self):
        self.__dataset = None
        self.__classes = None
        self.__data = None
    
    def __load_file(self, ROOT, FOLDER):
        
        dataset = GTZAN(root=ROOT, folder_in_archive=FOLDER)
        classes = np.unique([name.split('.')[0]for name in  dataset._walker])
        self.__classes = {c: torch.tensor(i) for i, c in enumerate(classes)}
        self.__classes_genre = { i : c for i, c in enumerate(classes)}
        self.__dataset = dataset
        self.__data = []
    
    def __crop(self, SIZE):
        for i in range(len(self.__dataset)):
            genre = self.__dataset[i][2]
            genre_onehot = one_hot(self.__classes[genre], num_classes=len(self.__classes)).type(torch.float32).to('cuda')
            signal = self.__dataset[i][0][:, :SIZE]
            self.__data.append((signal, genre_onehot))

    def load(self, args):
        try:
            with open('./pickle_data', 'rb') as pickle_in:
                dic = pickle.load(pickle_in)
                pickle_in.close()
                self.__classes = dic['classes']
                self.__classes_genre = dic['classes_genre']
            return dic['audio_data']
        except FileNotFoundError:
            self.__load_file(ROOT=args['ROOT'], FOLDER=args['FOLDER'])
            self.__crop(SIZE=args['SIZE'])

            audio_data = AudioDataset(data=self.__data)
            dic = {'audio_data': audio_data,'classes':self.__classes,"classes_genre":self.__classes_genre}
            with open('./pickle_data', 'wb') as pickle_out:
                pickle.dump(dic, pickle_out)
                pickle_out.close()
            return audio_data
    def classes(self):
        return self.__classes,self.__classes_genre