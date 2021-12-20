'''
random forest
'''

import enum
from numpy import e, nested_iters, printoptions
from numpy.lib.npyio import load
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.pyplot as plt


TRAIN_PATH = './Processed_Data/Integer_Data/training_full_label.csv'
TEST_PATH = './Processed_Data/Integer_Data/test_full_label.csv'


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

# n_estimators_lst = []
# score_lst = []
# for i in range(10):
#     n_estimators = 10+i*10
#     n_estimators_lst.append(n_estimators)
#     clf = RandomForestClassifier(n_estimators=n_estimators, max_features=4)
#     scores = cross_val_score(clf, X, y, cv=10)
#     score_lst.append(scores.mean())
#     print(scores.mean())

# # plt for selecting the best parameters
# plt.plot(n_estimators_lst, score_lst)
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.show()
# print(score_lst)

test_data = load_test_data()

# test k = n_estimators
clf = RandomForestClassifier(n_estimators=100, max_features=4)
clf.fit(X, y)
print('training end!')

predictions = clf.predict(test_data)


# store into the result
pre_res = [['ID', 'Prediction']]
for index, value in enumerate(predictions):
    pre_res.append([index+1, value])

with open('./Results/randomForest_100.csv', 'w') as f:
    csv_f = csv.writer(f, lineterminator='\n')
    csv_f.writerows(pre_res)