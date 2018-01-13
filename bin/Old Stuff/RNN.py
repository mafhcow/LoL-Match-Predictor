import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size = input_size, hidden_size = self.hidden_size,
                          num_layers = 1, nonlinearity = "relu", batch_first = True)
        self.h2o = nn.Linear(self.hidden_size, 2)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, input):
        (batch, _) = input.size()
        input = input.view(batch, 10, 139)
        _, hidden = self.rnn(input)
        hidden = self.dropout(hidden)
        # hidden is of dim layers x batch x hidden_size
        hidden = hidden.view(batch, self.hidden_size)

        output = self.softmax(self.h2o(hidden))

        return output
