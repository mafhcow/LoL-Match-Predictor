import time
import sys
sys.path.append('../utils')
import json
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import sklearn
import numpy as np
from DataBuilder import DataBuilder
from Models import SimpleNN, EmbeddedNN
    
builder = DataBuilder()
input_train, input_test, output_train, output_test = builder.build_data(embed=True)

model = SimpleNN(len(input_train[0]), 100, 0)
#model = EmbeddedNN(len(input_train[0]), 100, 500, 0)

criterion = nn.NLLLoss()

def train_model(train_data, test_data, model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 3e-5, weight_decay = 3e-6)
    prev_time = time.time()
    num_epochs = 10
    for epoch in range(num_epochs):
        print("****************************************")
        print("Epoch", epoch + 1)
        loss = run_epoch(train_data, True, model, optimizer)
        print("Trained in:", time.time() - prev_time)
        print("Loss:", loss)
        accuracy = run_epoch(test_data, False, model, optimizer)
        print("Tested in:", time.time() - prev_time)
        print("Accuracy:", accuracy)
        prev_time = time.time()
            

import math
        
def run_epoch(data, is_training, model, optimizer):
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        drop_last=False)
        
    if is_training:
        model.train()
    else:
        model.eval()

    losses = []
    true_labels = []
    for batch in data_loader:
        input = Variable(batch['input'])
        output = Variable(batch['output'])
        predictions = model(input)
        if is_training:
            loss = criterion(predictions, output)
            loss.backward()
            losses.append(loss.data[0])
            optimizer.step()
        else:
            def predict(out):
                predictions = []
                for i in range(out.shape[0]):
                    predictions.append(math.exp(out[i][1]))
                    #if out[i][0] > out[i][1]:
                    #    predictions.append(0)
                    #else:
                    #    predictions.append(1)
                return predictions
            guesses = predict(predictions.data.numpy())
            losses.extend(guesses)
            true_labels.extend(output.data.numpy())

    if is_training:
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        accuracy = sklearn.metrics.roc_auc_score(true_labels, np.array(losses))
        #accuracy = accuracy_score(np.array(losses), true_labels)
        return accuracy

train_data = [{'input': x, 'output': y} for (x,y) in zip(input_train, output_train)]
test_data = [{'input': x, 'output': y} for (x,y) in zip(input_test, output_test)]

train_model(train_data, test_data, model)





