
import torch.nn as  nn  
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4,4)
        self.fc2 = nn.Linear(4,3)
    def forward(self, x ):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


