import pytorch_lightning as pl
import torch
import torchmetrics

def main():
    print("START")
    print(pl.__version__)

    from net import Net
    pl.seed_everything(0)
    net = Net()
    trainer = pl.Trainer(max_epochs=30)
    #net = Net()
    #trainer = pl.Trainer(max_epochs=5)
    (train_loader,val_loader,test_dataloaders) = setup()
    trainer.fit(net ,train_loader , val_loader)

#    trainer.test(dataloaders=test_dataloaders)
    results = trainer.test()

def setup():
    from sklearn.datasets import load_iris
    from torch.utils.data import DataLoader , TensorDataset , random_split
    x,t = load_iris( return_X_y = True)
    x = torch.tensor( x , dtype= torch.float32)
    t = torch.tensor( t , dtype= torch.int64)

    dataset = TensorDataset(x,t)

    n_train = int(len(dataset)*0.6)
    n_val   = int(len(dataset)*0.2)
    n_test  = len(dataset) - n_train - n_val

    torch.manual_seed(0)
    train,val,test = random_split(dataset, [ n_train,n_val,n_test])
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train,batch_size, shuffle=True, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val,batch_size)
    test_loader = torch.utils.data.DataLoader(test ,batch_size)

    return ( train_loader  , val_loader,test_loader)


main()
