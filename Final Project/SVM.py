import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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


# c_range = []
# score_lst = []
# for i in range(20):
#     c = 1+i
#     c_range.append(c)
#     clf = make_pipeline(StandardScaler(), SVC(C=c, gamma='auto', kernel='rbf'))
#     scores = cross_val_score(clf, X, y, cv=10)
#     score_lst.append(scores.mean())
#     print(scores.mean())

# # plt for selecting the best parameters
# plt.plot(c_range, score_lst)
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.show()


clf = make_pipeline(StandardScaler(), SVC(C=2, gamma='auto', kernel='rbf'))
clf.fit(X, y)




# test_data = load_test_data()
# predictions = clf.predict(test_data)

# # store into the result
# pre_res = [['ID', 'Prediction']]
# for index, value in enumerate(predictions):
#     pre_res.append([index+1, value])

# with open('./Results/SVM_rbf.csv', 'w') as f:
#     csv_f = csv.writer(f, lineterminator='\n')
#     csv_f.writerows(pre_res)
