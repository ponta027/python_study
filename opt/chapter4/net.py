import pytorch_lightning as pl
import torch.nn as  nn
import torch.nn.functional as F
import torch
import tensorboard
import torchmetrics

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm1d(4)
        self.fc1    = nn.Linear(4,4)
        self.fc2    = nn.Linear(4,3)

        ##
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        ##

    def forward(self, x ):
        h = self.bn( x ) 
        h   = self.fc1(h)
        h   = F.relu(h) 
        h   = self.fc2(h)   
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


