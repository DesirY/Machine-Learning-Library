'''
This script is used to implement the missing data
'''
import csv
import numpy as np
from numpy import fabs, printoptions, random
from numpy.lib.function_base import percentile, sort_complex
from numpy.lib.shape_base import split
import pandas as pd

from BoostingTrees import process_training_data

MISSING_DATA = ['?']
CONTINUOUS_ATTR = ['fnlwgt']
INTEGER_ATTR = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']

def missing_data_handler(data):
    for col in data.columns:
        # print(col)
        rpl = ''
        if col in CONTINUOUS_ATTR:
            # average
            rpl = data[col].mean()
            data[col].fillna(rpl, inplace=True)
        elif col in INTEGER_ATTR:
            # median
            rpl = data[col].median()
            data[col].fillna(rpl, inplace=True)
        else:
            # mode
            rpl = data[col].mode()
            data[col].fillna(rpl[0], inplace=True)    # must rpl[0]
        
    return data

# split the training data into two sets, one is zero, one is one
training_data = pd.read_csv('./Data/train_final.csv', na_values=MISSING_DATA)
columns = training_data.columns

# complement the two separately
training_data_true = pd.DataFrame(columns=columns)
training_data_false = pd.DataFrame(columns=columns)

# traverse each row
for index, row in training_data.iterrows():
    if int(row['income>50K']) == 1:
        training_data_true = training_data_true.append(row, ignore_index=True)
    else:
        training_data_false = training_data_false.append(row, ignore_index=True)


# add missing data
training_data_true = missing_data_handler(training_data_true)
training_data_false = missing_data_handler(training_data_false)

# combine the two data
train_data = training_data_true.append(training_data_false, ignore_index=True)

train_data.to_csv('./Data/train_final_full.csv')