
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
    logger.debug("start main")
    logger.debug("start main")
    logger.debug(torch.__version__, torchvision.__version__)
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train = torchvision.datasets.MNIST(root='data',train = True, download=True, transform= transform)
    logger.debug(train)
    logger.debug("input:")
    logger.debug(train[0][0])
    logger.debug("input size:")
    logger.debug(train[0][0].shape)
    logger.debug("目標値")
    logger.debug(train[0][1])
    ##########################
    img = np.transpose(train[0][0],(1,2,0))
    logger.debug(img.shape)
    img = img.reshape(28,28)
    plt.gray()
    plt.imshow(img)
    plt.savefig('data/mnist_sample.jpg')
    plt.close()

def extract_feature_point():
    import torch.nn as nn 
    import torch.nn.functional as F
    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train = torchvision.datasets.MNIST(root='data',train = True, download=True, transform= transform)
    x= train[0][0]
    conv = nn.Conv2d(in_channels=1,out_channels=4, kernel_size=3,stride=1,padding=1)
    logger.debug(conv.weight)
    logger.debug(conv.weight.shape)
    x = x.reshape(1,1,28,28)
    x = conv(x)
    logger.debug(x)
    logger.debug(x.shape)
    x = F.max_pool2d( x ,kernel_size=2, stride=2)
    logger.debug(x.shape)
    x_shape = x.shape[1]*x.shape[2]*x.shape[3]
    x = x.view(-1,x_shape)
    logger.debug(x.shape)
    fc = nn.Linear( x_shape , 10 ) 
    x = fc(x)
    logger.debug(x)
    logger.debug(x.shape)

def cnn():
    logger.debug("start")
    import torch.nn as nn 
    import torch.nn.functional as F

    train_val = torchvision.datasets.MNIST( 
                root='data' , 
                train = True , 
                download=True,
                transform = transform
                )
    test = torchvision.datasets.MNIST( 
                root='data' , 
                train = False, 
                download=True,
                transform = transform
                )
    n_train = int(len(train_val)*0.8)
    n_val = len(train_val) - n_train
    torch.manual_seed(0)
    logger.debug("random_split")
    train,val = torch.utils.data.random_split( 
                train_val ,[n_train , n_val]
                )
    batch_size = 32

    logger.debug("dataloader")
    train_loader = torch.utils.data.DataLoader(
            train,
            batch_size,
            shuffle=True,
            drop_last = True
            )
    val_loader = torch.utils.data.DataLoader(
            val,
            batch_size
            )
    test_loader = torch.utils.data.DataLoader(
            test,
            batch_size
            )
    from net import Net
    pl.seed_everything(0)
    logger.debug("net")
    net = Net()
    logger.debug("Trainer")
    trainer = pl.Trainer( max_epochs = 10)
    logger.debug("fit")
    trainer.fit(net ,train_loader , val_loader )
    logger.debug("test")
    results =trainer.test( test_dataloaders=test_loader)
    logger.debug(results)
    
#main()
#extract_feature_point()
#
cnn()
logger.debug("end")
