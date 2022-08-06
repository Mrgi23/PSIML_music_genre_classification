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

if __name__ == '__main__':
        loader = Loader()
        args_loader = {'ROOT': './Data',
                       'FOLDER': 'genres_original',
                       'SIZE': 660000}
        audio = loader.load(args=args_loader)
        print('Loading done!\n')

        args_dataset = {'NFFT': 1024,
                        'SPLIT_PARTITIONS': 10}

        audio.preprocess(args=args_dataset)

        total_test_acc = []
        num_try = 10
        seeds = np.random.randint(1, 1024, num_try)
        for n, seed in enumerate(seeds):
                manual_seed(seed=seed)
                train_lenght = int(len(audio)*0.75)
                validation_lenght = int(len(audio)*0.15)
                test_lenght = len(audio) - train_lenght - validation_lenght
                train_dataset, validation_dataset, test_dataset = random_split(audio, lengths=[train_lenght,validation_lenght,test_lenght])

                # train_dataset = AudioDataset(train_dataset)
                # train_dataset.preprocess(args=args_dataset)

                # validation_dataset = AudioDataset(validation_dataset)
                # validation_dataset.preprocess(args=args_dataset)

                # test_dataset = AudioDataset(test_dataset)
                # test_dataset.preprocess(args=args_dataset)
                # print('Splitting done!\n')

                model = MusicModel()

                num_epochs = 1000
                batch_size = 250
                device = 'cuda'
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

                try:
                        with open('./pickle_weights', 'rb') as pickle_in:
                                dic = pickle.load(pickle_in)
                                pickle_in.close()
                        model.load_state_dict(dic['model_weights'])         
                except FileNotFoundError:
                        model.fit(args=args)
                        if n == 19:
                                dic = {'model_weights': model.state_dict()}
                                with open('./pickle_weights', 'wb') as pickle_out:
                                        pickle.dump(dic, pickle_out)
                                        pickle_out.close()
                print('Training done!\n')

                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                args = {'test_dataloader': test_dataloader,
                        'device': device}
                
                model.predict(args=args)
                total_test_acc.append(deepcopy(model.test_accuracy))
        fig = plt.figure(figsize=(16, 9))
        plt.xlabel('No. of trys')
        plt.ylabel('Accuracy [%]')
        plt.plot(np.arange(num_try), total_test_acc, label='Accuracy')
        plt.plot(np.arange(num_try), np.mean(total_test_acc)*np.ones(num_try), label='Mean value')
        plt.legend(loc='upper right')
        fig.savefig('CrossValidation.png')

        dic = {'seeds': seeds}
        with open('./pickle_seeds', 'rb') as pickle_in:
                dic = pickle.load(pickle_in)
                pickle_in.close()
                
