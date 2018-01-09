import time
import json
from xgboost import XGBClassifier
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.nn.functional import relu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.out = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, input):
        hidden = self.dropout(relu(self.linear(input)))
        out = self.out(hidden)
        out = self.softmax(out)
        return out
    
def map_champions_to_ids():
    champion_metadata = open('../Data/champion_metadata.json', 'r')
    text = champion_metadata.read()
    parsed = json.loads(text)

    mapping = {}
    for idx, champ in enumerate(parsed['champions']):
        mapping[int(champ['id'])] = idx
    return idx, mapping

def one_hot(champ_ids, max_length):
    vector = [0.0] * (max_length + 1)
    for champ_id in champ_ids:
        vector[champ_id] = 1.0
    return vector

def parse_line(line):
    blue_champs, red_champs, result = line.split('\t')
    blue_champs = [int(x) for x in blue_champs.split(' ')]
    red_champs = [int(x) for x in red_champs.split(' ')]
    result = int(result)
    return blue_champs, red_champs, result

def build_one_hot_data():
    with open('../Data/champions.txt', 'r') as f:
        max_id, mapping = map_champions_to_ids()
        inputs = []
        outputs = []
        for line in f.readlines():
            blue_champs, red_champs, result = parse_line(line)
            blue_vector = one_hot([mapping[x] for x in blue_champs], max_id)
            red_vector = one_hot([mapping[x] for x in red_champs], max_id)
            inputs.append(blue_vector + red_vector)
            outputs.append(result)
        return np.array(inputs), np.array(outputs)
        
inputs, outputs = build_one_hot_data()
input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size = 0.25)

#classifier = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
#classifier.fit(input_train, output_train)
#predictions = classifier.predict(input_test)
#print(classification_report(output_test, predictions, digits=4))
#print('Accuracy for random forest: ' + str(accuracy_score(output_test, predictions)))

#classifier = XGBClassifier()
#classifier.fit(input_train, output_train)
#predictions = classifier.predict(input_test)
#print(classification_report(output_test, predictions, digits=4))
#print('Accuracy for xgboost: ' + str(accuracy_score(output_test, predictions)))

#classifier = SVC()
#classifier.fit(input_train, output_train)
#predictions = classifier.predict(input_test)
#print(classification_report(output_test, predictions, digits=4))
#print('Accuracy for SVM: ' + str(accuracy_score(output_test, predictions)))

input_train = torch.FloatTensor(input_train)
input_test = torch.FloatTensor(input_test)
output_train = torch.LongTensor(output_train)
output_test = torch.LongTensor(output_test)


model = SimpleNN(len(input_train[0]), 100)
criterion = nn.NLLLoss()


def train_model(train_data, test_data, model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr= 1e-5, weight_decay = 1e-6)
    prev_time = time.time()
    for epoch in range(50):
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
