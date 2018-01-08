import time
import json
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import random
from torch.autograd import Variable
from torch.nn.functional import relu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from EmbeddingNN import EmbeddingNN
from RNN import RNN
import torch.nn.functional as F

gpu = True
#dataPath = "../Data/champions.txt"
dataPath = "../Data/drafted_champions.txt"

def readData(dataPath, draftOrder = False):
    with open(dataPath) as f:
        x = []
        for line in f:
            contents = line.strip().split("\t")
            if not(draftOrder):
                x.append((contents[0].split(" "), contents[1].split(" "), contents[2]))
            else:
                x.append((contents[0].split(" "), contents[1]))
    return x

def getIdMapping(rawData, draftOrder = False):
    champIds = {}
    index = 0

    for i in range(len(rawData)):
        if not(draftOrder):
            data = rawData[i][0] + rawData[i][1]
        else:
            data = rawData[i][0]
            
        for d in data:
            if d in champIds:
                continue
            else:
                champIds[d] = index
                index += 1

    return champIds, index

def convert(champId, champIds):
    array = np.zeros(139)
    array[champIds[champId]] = 1.0
    return array

def generateTrainTest(rawData, champIds, split = 0.25, draftOrder = False):
    X = []
    Y = []

    Xtest = []
    Ytest = []
    
    for i in range(len(rawData)):
        if not(draftOrder):
            champs = rawData[i][0] + rawData[i][1]
        else:
            champs = rawData[i][0]
        tmp = []
        for j in range(10):
            tmp.append(convert(champs[j], champIds))

        outcome = int(rawData[i][2 - draftOrder])
        if random.random() < 1.0 - split:
            X.append(np.concatenate(tmp))
            Y.append(outcome)
        else:
            Xtest.append(np.concatenate(tmp))
            Ytest.append(outcome)

    return X, Y, Xtest, Ytest


def train_model(train_data, test_data, model, gpu = False):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 1e-5, weight_decay = 1e-6)
    prev_time = time.time()

    if gpu:
        model.cuda()

    for epoch in range(50):
        print("****************************************")
        print("Epoch", epoch + 1)
    
        loss = run_epoch(train_data, True, model, optimizer, gpu = gpu)
        print("Trained in:", time.time() - prev_time)
        print("Loss:", loss)
        accuracy = run_epoch(test_data, False, model, optimizer, gpu = gpu)
        print("Tested in:", time.time() - prev_time)
        print("Accuracy:", accuracy)
        prev_time = time.time()
        torch.save(model, "models/model{}".format(epoch))
        
import math

def run_epoch(data, is_training, model, optimizer, gpu = False):
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=50,
        shuffle=True,
        num_workers=4,
        drop_last=False)
    
    losses = []
    true_labels = []

    if is_training:
        model.train()
    else:
        model.eval()

    for batch in data_loader:
        input = Variable(batch['input'])
        output = Variable(batch['output'])

        if gpu:
            input = input.cuda()
            output = output.cuda()
            
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
            guesses = predict(predictions.cpu().data.numpy())
            losses.extend(guesses)
            true_labels.extend(output.cpu().data.numpy())
                
    if is_training:
        avg_loss = np.mean(losses)
        return avg_loss
    else:
        accuracy = sklearn.metrics.roc_auc_score(true_labels, np.array(losses))
        #accuracy = accuracy_score(np.array(losses), true_labels)
        return accuracy

draftOrder = "drafted" in dataPath
rawData = readData(dataPath, draftOrder = draftOrder)
champIds, index = getIdMapping(rawData, draftOrder = draftOrder)
X, Y, Xtest, Ytest = generateTrainTest(rawData, champIds, draftOrder = draftOrder)
    
input_train = torch.FloatTensor(X)
input_test = torch.FloatTensor(Xtest)
output_train = torch.LongTensor(Y)
output_test = torch.LongTensor(Ytest)

if not(draftOrder):
    model = EmbeddingNN(len(input_train[0]), 300, 0.2, 300)
else:
    model = RNN(139, 1000, 0.2)
criterion = nn.NLLLoss()

train_data = [{'input': x, 'output': y} for (x,y) in zip(input_train, output_train)]
test_data = [{'input': x, 'output': y} for (x,y) in zip(input_test, output_test)]

train_model(train_data, test_data, model, gpu = gpu)
