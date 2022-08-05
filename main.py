from preprocessing import Preprocessing
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import random_split
from model import MusicModel
from torch.nn import CrossEntropyLoss
import pickle
if __name__ == '__main__':
        preproc = Preprocessing()
        args = {'ROOT': './Data',
                'FOLDER': 'genres_original',
                'NFFT': 1024,
                'SIZE': 660000,
                'SPLIT_PARTITIONS': 10}
        audio,dataset = preproc.preprocessing(args=args)

        train_lenght = int(len(dataset)*0.75)
        validation_lenght = int(len(dataset)*0.15)
        test_lenght = len(dataset) - train_lenght - validation_lenght
        train_dataset, validation_dataset, test_dataset = random_split(dataset,lengths=[train_lenght,validation_lenght,test_lenght])
        print('Splitting done!\n')

        model = MusicModel()

        num_epochs = 1000
        batch_size = 50
        device = 'cuda'
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        loss_func = CrossEntropyLoss()
        lr = 1e-4
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
                print('Loading done!\n')
                model.fit(args=args)
                dic = {'model_weights': model.state_dict()}
                with open('./pickle_weights', 'wb') as pickle_out:
                        pickle.dump(dic, pickle_out)
                        pickle_out.close()
        print('Training done!\n')
                
