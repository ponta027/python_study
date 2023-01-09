import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

transform = transforms.Compose([
        transforms.ToTensor()
    ])


############ LOGGER

from logging import getLogger, StreamHandler, DEBUG,Formatter
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
rh_formatter = Formatter('%(asctime)s : %(levelname)s - %(filename)s - %(message)s')
handler.setFormatter(rh_formatter)
logger.addHandler(handler)
logger.propagate = False


def main():
    logger.debug('start')

    train_val = torchvision.datasets.CIFAR10(root='data', train= True, download=True,
            transform=transform)
    test    = torchvision.datasets.CIFAR10(root='data', train= False, download=True,
            transform=transform)

    n_train = int(len(train_val)*0.8)
    n_val = len(train_val)-n_train

    torch.manual_seed(0)
    logger.debug('random_split')
    train,val = torch.utils.data.random_split(train_val,[n_train,n_val])
    batch_size = 256

    logger.debug('dataloader')
    train_loader = torch.utils.data.DataLoader( train ,batch_size, shuffle=True, drop_last=True)
    val_loader=torch.utils.data.DataLoader( val ,batch_size)
    test_loader=torch.utils.data.DataLoader( test ,batch_size)

    pl.seed_everything(0)
    from net_cifar10 import Net
    logger.debug('net')
    net=Net()
    logger.debug('Trainer')
    trainer = pl.Trainer( max_epochs=10)
    #trainer = pl.Trainer( max_epochs=10,gpus=1)
    logger.debug('Trainer.fit')
    trainer.fit( net ,train_loader, val_loader)
    logger.debug('Trainer.test')
    results = trainer.test(test_dataloaders=test_loader)
    print(results)
    logger.debug('Finished')

if __name__ == '__main__':
    main()
