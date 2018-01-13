import sys 	
sys.path.append('../utils')
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from DataBuilder import DataBuilder
    
builder = DataBuilder()
input_train, input_test, output_train, output_test = builder.build_data(embed=False)

classifier = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
classifier.fit(input_train, output_train)
predictions = classifier.predict(input_test)
print(classification_report(output_test, predictions, digits=4))
print('Accuracy for random forest: ' + str(accuracy_score(output_test, predictions)))

#classifier = XGBClassifier(n_estimators = 200)
#classifier.fit(input_train, output_train)
#predictions = classifier.predict(input_test)
#print(classification_report(output_test, predictions, digits=4))
#print('Accuracy for xgboost: ' + str(accuracy_score(output_test, predictions)))

#classifier = SVC()
#classifier.fit(input_train, output_train)
#predictions = classifier.predict(input_test)
#print(classification_report(output_test, predictions, digits=4))
#print('Accuracy for SVM: ' + str(accuracy_score(output_test, predictions)))