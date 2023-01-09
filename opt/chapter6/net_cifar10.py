
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn 
import torch
import torch.nn.functional as F

class Net(pl.LightningModule):


    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d( in_channels=3,out_channels=6,kernel_size=(3,3),padding=(1,1))
        self.fc    = nn.Linear(1536,10)

        ##
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        ##

    def forward(self, x ):
        h = self.conv( x ) 
        h= F.max_pool2d( h , kernel_size=(2,2) , stride =2)
        h   = F.relu(h) 
        h = h.view(-1,1536)
        h   = self.fc(h)   
        return h

    def training_step(self,batch,batch_idx):
        x,t = batch
        y = self(x)
        loss = F.cross_entropy( y ,t )
        
        #########
        self.log('train_loss' , loss , on_step =True, on_epoch=True)
        self.log('train_acc' , self.train_acc(y,t), on_step =True, on_epoch=True)
        #########
        return loss

    def validation_step(self,batch,batch_idx):
        x,t = batch
        y = self(x)
        loss = F.cross_entropy( y ,t )
        
        #########
        self.log('val_loss' , loss , on_step =False, on_epoch=True)
        self.log('val_acc' , self.val_acc(y,t), on_step =False, on_epoch=True)
        #########
        return loss

    def test_step(self,batch,batch_idx):
        x,t = batch
        y = self(x)
        loss = F.cross_entropy( y ,t )
        
        #########
        self.log('test_loss' , loss , on_step =False , on_epoch=True)
        self.log('test_acc' , self.test_acc(y,t), on_step =False , on_epoch=True)
        #########
        return loss




    def configure_optimizers(self):
        optimizer = torch.optim.SGD( self.parameters() , lr = 0.01)
        return optimizer


