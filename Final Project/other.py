import enum
from numpy import e, nested_iters, printoptions
from numpy.lib.npyio import load
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

TRAIN_PATH = './Processed_Data/one_hot_Data/training_one_hot.csv'
TEST_PATH = './Processed_Data/one_hot_Data/test_one_hot.csv'


def load_training_data():
    X = []
    y = []

    with open(TRAIN_PATH, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for line in f_csv:
            X.append(line[:-1])
            y.append(line[-1])

    return (X, y)


def load_test_data():
    res = []
    with open(TEST_PATH, 'r') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for line in f_csv:
            res.append(line)
    return res



X, y = load_training_data()



clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)

print(clf.score(X, y))

test_data = load_test_data()
predictions = clf.predict(test_data)

# store into the result
pre_res = [['ID', 'Prediction']]
for index, value in enumerate(predictions):
    pre_res.append([index+1, value])

with open('./Results/logistic.csv', 'w') as f:
    csv_f = csv.writer(f, lineterminator='\n')
    csv_f.writerows(pre_res)

