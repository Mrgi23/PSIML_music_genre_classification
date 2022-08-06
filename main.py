from copy import deepcopy
from dataset import AudioDataset
from loader import Loader
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from model import MusicModel
from torch.nn import CrossEntropyLoss
from torch import manual_seed
import pickle
from matplotlib import pyplot as plt
import numpy as np
from torch import Generator
import os
if __name__ == '__main__':
        loader = Loader()
        args_loader = {'ROOT': './Data',
                       'FOLDER': 'genres_original',
                       'SIZE': 660000}
        audio = loader.load(args=args_loader)
        classes, classes_genre = loader.classes()
        print('Loading done!\n')

        args_dataset = {'NFFT': 1024,
                        'SPLIT_PARTITIONS': 10}

        audio.preprocess(args=args_dataset)

        total_test_acc = []
        best_models = []
        num_try = 5
        seeds = np.arange(0,num_try)
        for n, seed in enumerate(seeds):
                manual_seed(seed=seed)
                train_lenght = int(len(audio)*0.75)
                validation_lenght = int(len(audio)*0.15)
                test_lenght = len(audio) - train_lenght - validation_lenght
                train_dataset, validation_dataset, test_dataset = random_split(audio, lengths=[train_lenght,validation_lenght,test_lenght], 
                                                                                generator=Generator().manual_seed(int(seed)))
                # train_dataset = AudioDataset(train_dataset)
                # train_dataset.preprocess(args=args_dataset)

                # validation_dataset = AudioDataset(validation_dataset)
                # validation_dataset.preprocess(args=args_dataset)

                # test_dataset = AudioDataset(test_dataset)
                # test_dataset.preprocess(args=args_dataset)
                print('Splitting done!\n')


                num_epochs = 1000
                batch_size = 50
                device = 'cuda'
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                data_used = list(next(iter(train_dataloader))[0].keys())
                model = MusicModel(['spectrogram','mfcc'])

                loss_func = CrossEntropyLoss()
                lr = 5e-5
                optimizer = Adam(model.parameters(), lr=lr)
                args = {'num_epochs': num_epochs,
                        'device': device,
                        'train_dataloader': train_dataloader,
                        'validation_dataloader': validation_dataloader,
                        'loss_func': loss_func,
                        'optimizer': optimizer}

                model = model.to(device=device)
                if os.path.exists('./pickle_weights'):
                        with open('./pickle_weights', 'rb') as pickle_in:
                                state = pickle.load(pickle_in)
                        model.load_state_dict(state)         
                else:
                        model.fit(args=args)
                        print('Training done!\n')


                test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0)
                args = {'test_dataloader': test_dataloader,
                        'device': device,
                        'seed':seed,
                        'classes':classes,
                        'classes_genre':classes_genre}
                
                model.predict(args=args)
                best_models.append(model.state_dict())
                total_test_acc.append(model.test_accuracy)
        with open('./pickle_weights', 'wb') as pickle_out:
                pickle.dump(best_models[np.argmax(total_test_acc)], pickle_out)
        fig = plt.figure(figsize=(16, 9))
        plt.xlabel('No. of trys')
        plt.ylabel('Accuracy [%]')
        plt.xticks(np.arange(1,num_try+1))
        plt.plot(np.arange(1, num_try+1), total_test_acc, label='Accuracy')
        plt.plot(np.arange(1, num_try+1), np.mean(total_test_acc)*np.ones(num_try), label='Mean value')
        plt.legend(loc='upper right')
        fig.savefig('CrossValidation.png')

                
