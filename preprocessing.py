from locale import normalize
import scipy as sp
from scipy.io.wavfile import read
from scipy.signal import spectrogram
import numpy as np
import torch

class Preprocessing():
    
    def __init__(self):
        self.__fs = None
        self.__signal = None
        self.__signal_split = None
        self.__spectrogram = None
    
    def __load_file(self, PATH, NORMALIZE=False):
        self.__fs, self.__signal = read(PATH)
        
        if NORMALIZE:
          m = np.tile(np.mean(self.__signal), self.__signal.shape[0])
          s = np.tile(np.std(self.__signal), self.__signal.shape[0])
          self.__signal = (self.__signal - m) / s
    
    def __split(self, SPLIT_PARTITIONS):
        dim = int(self.__signal.shape[0]//SPLIT_PARTITIONS)
        crop = self.__signal.shape[0] - SPLIT_PARTITIONS*dim
        if crop == 0:
            pass

        elif crop % 2:
            self.__signal = self.__signal[crop//2:-(crop//2+1)]
        else:
            self.__signal = self.__signal[crop//2:-crop//2]
            
        self.__signal_split = np.zeros((SPLIT_PARTITIONS, dim))
        for i in range(SPLIT_PARTITIONS):
            self.__signal_split[i] = (self.__signal[i*dim:(i+1)*dim])
    
    def __spectrograms(self, NFFT,  OVERLAP):
        N = int(self.__signal_split.shape[0])
        M = int(self.__signal_split.shape[1])
        X = int(1 + NFFT//2)
        Y = int(1 + (M-NFFT)//((1-OVERLAP)*NFFT))
        self.__spectrogram = np.zeros((N, X, Y))
        for i in range(N):
            self.__spectrogram[i] = spectrogram(x=self.__signal_split[i],
                                                          fs=self.__fs,
                                                          nfft=NFFT,
                                                          nperseg=NFFT,
                                                          noverlap=(OVERLAP*NFFT))[-1]
    
    def preprocessing(self, args):
        self.__load_file(PATH=args['PATH'], NORMALIZE=args['NORMALIZE'])
        self.__split(SPLIT_PARTITIONS=args['SPLIT_PARTITIONS'])
        self.__spectrograms(NFFT=args['NFFT'], OVERLAP=args['OVERLAP'])
        
    @property
    def get_signals(self):
        if self.__signal_split is not None:
            return torch.from_numpy(self.__signal_split)
        else:
            return None
    
    @property
    def get_spectrograms(self):
        if self.__spectrogram is not None:
            return torch.from_numpy(self.__spectrogram)
        else:
            return None