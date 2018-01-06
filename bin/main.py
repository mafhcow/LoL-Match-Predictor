import time
import json
import sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metric import accuracy_score

def map_champions_to_ids():
    champion_metadata = open('../Data/champion_metadata.json', 'r')
    text = champion_metadata.read()
    parsed = json.loads(text)

    mapping = {}
    for idx, champ in enumerate(parsed['champions']):
        mapping[int(champ['id'])] = idx
    return idx, mapping

def one_hot(champ_ids, max_length):
    vector = [0] * (max_length + 1)
    for champ_id in champ_ids:
        vector[champ_id] = 1
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

classifier = RandomForestClassifier()
classifier.fit(input_train, output_train)
predictions = classifier.predict(input_test)
print(classification_report(output_test, predictions, digits=4))
print('Accuracy for random forest': accuracy_score(output_test, predictions))

##classifier = SVC()
##classifier.fit(input_train, output_train)
##predictions = classifier.predict(input_test)
##print(classification_report(output_test, predictions, digits=4))
##print('Accuracy for SVM': accuracy_score(output_test, predictions))
