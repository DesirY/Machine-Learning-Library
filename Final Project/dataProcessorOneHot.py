'''
one hot encoding
'''

import csv
import numpy as np
import pandas as pd
from pandas.io.stata import PossiblePrecisionLoss

training_data = pd.read_csv('./Processed_Data/training_data_full.csv')
training_X = training_data.iloc[:, 0:-1]
training_y = training_data.iloc[:, -1]

# dummy for training_X
training_X = pd.get_dummies(training_X)

# contact
training_X.append(training_y)

training_result = training_X.join(training_y)

training_result.to_csv('./Processed_Data/one_hot_Data/training_full_one_hot.csv')

test_data = pd.read_csv('./Processed_Data/test_data_full.csv')
test_data = pd.get_dummies(test_data)
test_data.to_csv('./Processed_Data/one_hot_Data/test_full_one_hot.csv')
