''''
test methods of np
'''
import numpy as np
from numpy.core.arrayprint import printoptions
import math

arr = np.array([[1, 2, 3, 4]])
arr.shape = (4, 1)

s = np.sum(arr**2)

# print(arr)
# print(s)
# print(arr.shape)

def sigmoid(a):
    return (1.0/(1.0+np.exp(-a)))

print((sigmoid(-0.135)-1)*3)