import torch
import warnings
import torch.nn as  nn  

warnings.filterwarnings('ignore')



torch.__version__

print("重回帰分析の線形結合")
fc = nn.Linear(3,2)
print(fc)
print("W:重み、b :バイアス")
print(fc.weight)
print(fc.bias)

print("================")
print("乱数シードの固定")
torch.manual_seed(1)
fc = nn.Linear(3,2)
print(fc)
print("W:重み、b :バイアス")
print(fc.weight)
print(fc.bias)

print("線形変換")
x = torch.tensor([[1,2,3]]  , dtype=torch.float32)
print(x)
print(type(x))
print("線形変換 (1,3) * (3,2)=(1,2)")
u = fc(x)
print(u)

print("""活性化関数ReLU""")
import torch.nn.functional as F
h = F.relu(u)
print(h)

torch.manual_seed(1)
print("(1,3)*(3,2)*ReLU*(2,1)=(1,1)")

x = torch.tensor([[1,2,3]],dtype=torch.float32)
fc1 = nn.Linear(3,2)    # element 3 to 2
fc2 = nn.Linear(2,1)    # element 2   to  1

u1=fc1(x)
h1  =   F.relu(u1)

y   =   fc2(h1)
print(y)


#############################
# loss function
#############################
t = torch.tensor([[1]],dtype=torch.float32)
print(t)
loss = F.mse_loss(t,y)
print(loss)

print("""
3.3 学習

Iris データの読み出し。 load_iris

ID,変数名,説明,データ型
0,  sepa_length,    がく片の長さ    , float
1,  sepa_width  ,   がく片の幅      ,   float
2,  petal_length,   花びらの長さ    ,   float
3,  petal_width,    花びらの幅      ,   float
        """)


from sklearn.datasets   import load_iris
print("Load DataSet: x:入力値、目標値:t")
x,t = load_iris(return_X_y=True)
print(x.shape,t.shape)
print(type(x),type(t))
print(x.dtype,t.dtype)

print("Convert torch.Tensor Type")
x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

# test data
#print(x,t)

print("""
学習時に使用するデータx,tを一つのデータオブジェクトにまとめる。
        """)
from torch.utils.data import TensorDataset
dataset = TensorDataset(x,t)
print(dataset)
print(type(dataset))

print("(入力変数、教師データ)")
print(dataset[0])
print("Len:{}".format(len(dataset)))


print("""
# 各データセットのサンプル数を決定
        """)
# 各データセットのサンプル数を決定

n_train = int((len(dataset)*0.6))
n_val = int(len(dataset)*0.2)
n_test  = len(dataset) -n_train -   n_val
print("train:{},val:{},test:{}".format(n_train,n_val,n_test))


from torch.utils.data import random_split

torch.manual_seed(0)
train , val , test = random_split(dataset,[n_train, n_val,n_test])

print("train:{},va:{},test:{}".format(len(train),  len(val)    ,   len(test)))


print("""
ミニバッチ学習
        """)


batch_size = 10
from torch.utils.data import DataLoader

print("バッチサイズに分割")
train_loader = DataLoader( train ,batch_size , shuffle=True)
val_loader = DataLoader( val ,batch_size , shuffle=True)
test_loader = DataLoader( test ,batch_size , shuffle=True)

print(train_loader)


print("""
ネットワークを学習
* fc1 : input 4 =>  output:4
* fc2 : input 4 =>  output:3

順伝播の計算：forward
    * fc1 ->ReLU -> fc2 -> softmax
        """)

from net    import Net
torch.manual_seed(0)
net = Net()
print(net)

print(""" 損損失関数を選択
3クラスの分類のため、損失関数としてクロスエントロピーを採用
L = -Sum_{n=1}^{N}Sum_{k=1}^{K}t_{n,k}logy_{n,k}

        """)
criterion = nn.CrossEntropyLoss()
print(criterion)

print("""最適化手法を選択
確率的勾配降下法(SGD)

ネットワークの学習にあたり使用する最適化手法を選択
パラメータのの取得には、net.parameter()を用いる。
学習係数も設定する。

        """)


print(net.parameters())
optimizer = torch.optim.SGD( net.parameters() ,lr=0.1)

print("""
ネットワークを学習
1. ミニバッチ単位でサンプルX,tを抽出
2.  現在のパラメータW,bを利用して、順伝播で予測値yを算出
3.  目標値tと予測値yから損失関数Lを算出
4.  誤差逆伝播法に基づいて各パラメータの勾配を算出
5.  勾配の値に基づいて選択した最適化手法によりパラメータW,bを更新
        """)


for batch   in  train_loader:
    batch = next((iter(train_loader)))
    x,t=    batch
#    print("fc1:Weight:{},Bias:{}".format(net.fc1.weight,net.fc1.bias))
#    print("fc2:Weight:{},Bias:{}".format(net.fc2.weight,net.fc2.bias))
    y = net.forward(x)
    print(""" result of foward""")
    print(y)
    y = net(x)
    # calculate loss function
    loss = criterion(y,t)

