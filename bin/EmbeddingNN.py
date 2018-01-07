import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingNN(nn.Module):
    def __init__(self, input_size, hidden, dropout, embedding_dim):
        super(EmbeddingNN, self).__init__()
        self.linear = nn.Linear(embedding_dim * 10, hidden)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p = dropout)
        self.conv = nn.Conv1d(1, embedding_dim, 139, stride=139)
        self.embedding_dim = embedding_dim
    
    def forward(self, input):
        (batch, _) = input.size()
        input = input.view(batch, 1, 1390)
        convlayer = self.dropout(self.conv(input))
        convlayer = torch.transpose(convlayer, 1, 2)
        convlayer = convlayer.contiguous().view(batch * 10, self.embedding_dim)
        convlayer = self.dropout(F.relu(self.linear2(convlayer)))
        convlayer = convlayer.view(batch, self.embedding_dim * 10)
        
        hidden = self.dropout(F.relu(self.linear(convlayer)))
        #hidden = self.dropout(relu(self.linear2(hidden)))
        out = self.out(hidden)
        out = self.softmax(out)
        return out
