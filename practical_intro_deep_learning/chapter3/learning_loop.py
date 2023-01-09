import torch
import warnings
import torch.nn as  nn  
from sklearn.datasets   import load_iris
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from net    import Net

warnings.filterwarnings('ignore')

def load_data():
    x,t = load_iris(return_X_y=True)

    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.int64)

    dataset = TensorDataset(x,t)
    return dataset

def load_data_batch( dataset ):
    n_train = int((len(dataset)*0.6))
    n_val = int(len(dataset)*0.2)
    n_test  = len(dataset) -n_train -   n_val

    torch.manual_seed(0)
    train , val , test = random_split(dataset,[n_train, n_val,n_test])

    batch_size = 10

    train_loader = DataLoader( train ,batch_size , shuffle=True)
    val_loader = DataLoader( val ,batch_size , shuffle=True)
    test_loader = DataLoader( test ,batch_size , shuffle=True)

    return (train_loader,val_loader,test_loader)


dataset = load_data()
(train_loader, val_loader , test_loader ) = load_data_batch(dataset)

device = torch.device('cuda:0' if torch.cuda.is_available()    else 'cpu')

torch.manual_seed(0)
net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( net.parameters() ,lr=0.1)

max_epoch = 1


"""
for epoch in range(max_epoch):
    for batch   in  train_loader:
        batch = next((iter(train_loader)))
        x,t=    batch
        # transfer to device
        x   = x.to(device)
        t   = t.to(device)

        optimizer.zero_grad()
        y = net(x)
        # calculate loss function
        loss = criterion(y,t)

        # 正解率
        y_label = torch.argmax(y,dim=1)
        acc = (y_label==t).sum() * 1.0/len(t)

        print("accuracy:", acc )

#        print('loss:',loss.item())

        loss.backward()
        optimizer.step()
"""

def calc_acc( data_loader ):
    with torch.no_grad():
        accs =[]
        for batch in data_loader:
            x,t=batch
            x   = x.to(device)
            t   = t.to(device)
            y = net(x)
            y_label = torch.argmax(y , dim=1)
            acc = (y_label ==t ).sum() * 1.0/len(t)
            accs.append(acc)
    avg_acc = torch.tensor(accs).mean()
    return avg_acc

val_acc = calc_acc( train_loader)
print(val_acc)

test_acc = calc_acc( test_loader)
print(test_acc)
"""
val_acc = calc_acc( val_loader)
print(val_acc)
"""

