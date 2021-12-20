'''
label encoder
'''
import numpy as np
from numpy.core.arrayprint import printoptions
import pandas as pd
from sklearn.preprocessing import LabelEncoder

training_data = pd.read_csv('./Processed_Data/training_data_full.csv')
training_X = training_data.iloc[:, 0:-1]
training_y = training_data.iloc[:, -1]

X_cols = training_X.columns.values

training_X = training_X.apply(LabelEncoder().fit_transform)

training_result = training_X.join(training_y)

training_result.to_csv('./Processed_Data/Integer_Data/training_full_label.csv')

test_data = pd.read_csv('./Processed_Data/test_data_full.csv')
test_data = test_data.apply(LabelEncoder().fit_transform)
test_data.to_csv('./Processed_Data/Integer_Data/test_full_label.csv')