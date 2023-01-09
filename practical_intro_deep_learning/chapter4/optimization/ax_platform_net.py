from net_opt import Net
import pytorch_lightning as pl
import torch
import torchmetrics
import ax


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


def non_opt():
    train_loader , val_loader ,test_loader = setup()
    pl.seed_everything(0)
    net = Net()
    trainer = pl.Trainer(max_epochs = 10)
    trainer.fit(net , train_loader ,val_loader)

def opt():
    """ 
    accuracyが最大化するような最適化を行う。
    """
    train_loader , val_loader ,test_loader = setup()
    pl.seed_everything(0)
    net = Net()
    trainer = pl.Trainer(max_epochs = 10)
    trainer.fit(net , train_loader ,val_loader)
    metrics = trainer.callback_metrics['val_acc']
    print(metrics)

def optimization():
    parameters = [
            {'name':'n_mid','type':'range','bounds':[1,100]},
            {'name':'lr','type':'range','bounds':[0.001,0.1]}
            ]
    results = ax.optimize( parameters  , evaluation_function ,  random_seed=0)
    (best_parameters , values, experiment , model ) = results
    print(best_parameters)

def evaluation_function( parameters ):
    n_mid = parameters.get('n_mid')
    lr = parameters.get('lr')
    torch.manual_seed(0)
    net = Net(n_mid = n_mid , lr =  lr )
    #net = Net(batch_size=10, n_mid = n_mid , lr =  lr )
    trainer =  pl.Trainer( show_progress_bar =  False)
    trainer.fit(net)

    val_acc  = trainer.callback_mmetrics['val_acc']
    trainer.test()
    test_acc = trainer.callback_metrics['test_acc']

    print('n_mid:' , n_mid ) 
    print(  'lr:'   ,   lr)
    print(  'val_acc:'  ,   val_acc)
    print(  'test__acc:',   test_acc)
    
    return val_acc


optimization()
#setup()
#opt()


