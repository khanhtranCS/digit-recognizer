import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

arr_train = df_train.values;
arr_test = df_test.values;

train_rows, train_cols = arr_train.shape;
test_rows, test_cols = arr_test.shape;

train_label = arr_train[:, :1]
train_digits= arr_train[:train_rows, 1:]

test_digits = arr_test[:test_rows, :]

#print(test_digits.shape)
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def predictRandForest(train, labels, test):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=2)
    rf.fit(train, labels)
    rf_predictions = rf.predict(test)
    rf_probs = rf.predict_proba(test)
    rf_BestProbs = rf_probs.max(axis=1)
    return rf_predictions, rf_BestProbs

rfPredictions, rfScore = predictRandForest(train_digits, train_label, test_digits)

for i in range(len(rfPredictions)):
    print(rfPredictions[i])
    print(rfScore[i])
