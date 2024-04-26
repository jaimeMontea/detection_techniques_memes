import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple classification model using MLP

class BaseBinaryMLP(nn.Module):
    def _init_(self,input_size,layer1_size,layer2_size,dropout):
        super()._init_()
        self.linear1 = nn.Linear(input_size,layer1_size)
        self.linear2 = nn.Linear(layer1_size,layer2_size)
        self.linear3 = nn.Linear(layer2_size,1)
        self.dropout = dropout
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.dropout(x,self.dropout)
        x = F.relu(self.linear2(x))
        x = F.dropout(x,self.dropout)
        return self.linear3(x)