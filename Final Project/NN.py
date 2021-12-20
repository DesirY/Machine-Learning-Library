import numpy as np
from numpy import printoptions, random
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import csv

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

hidden_num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
nureon_num = [150, 200, 300, 450]
score_lst = []

for i in range(10):
    hidden = hidden_num[i]
    clf = MLPClassifier(random_state= 400,  hidden_layer_sizes= hidden, )
    scores = cross_val_score(clf, X, y, cv=10)
    score_lst.append(scores.mean())
    print(scores.mean())


plt.plot(hidden_num, score_lst)
plt.xlabel('hidden_layer_sizes')
plt.ylabel('Accuracy')
plt.show()

# print(score_lst)

# clf = MLPClassifier(random_state= 400,  hidden_layer_sizes= 4)
# clf.fit(X, y)

# test_data = load_test_data()
# predictions = clf.predict(test_data)

# # store into the result
# pre_res = [['ID', 'Prediction']]
# for index, value in enumerate(predictions):
#     pre_res.append([index+1, value])

# with open('./Results/nn_.csv', 'w') as f:
#     csv_f = csv.writer(f, lineterminator='\n')
#     csv_f.writerows(pre_res)
