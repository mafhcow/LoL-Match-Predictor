import sys
sys.path.append('../utils')
import torch
import torch.nn as nn
from torch.nn.functional import relu
from mappings import num_champs

NUM_CHAMPS = num_champs()

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(SimpleNN, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, input):
        hidden = self.dropout(relu(self.i2h(input)))
        out = self.h2o(hidden)
        out = self.softmax(out)
        return out

class EmbeddedNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout):
        self.input_size = input_size
        super(EmbeddedNN, self).__init__()
        self.embed = nn.Linear(int(input_size/10), embedding_size)
        self.e2h = nn.Linear(10*embedding_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, 2)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, input):
        hiddens = []
        hidden_inputs = torch.split(input, split_size=int(self.input_size/10), dim=1)
        for inp in hidden_inputs:
            hiddens.append(self.dropout(relu(self.embed(inp))))
        combined = torch.cat((hiddens[0], hiddens[1], hiddens[2], hiddens[3], hiddens[4], hiddens[5], hiddens[6], hiddens[7], hiddens[8], hiddens[9]), 1)
        hidden = self.dropout(relu(self.e2h(combined)))
        out = self.h2o(hidden)
        out = self.softmax(out)
        return out