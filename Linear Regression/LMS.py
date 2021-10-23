from os import X_OK
from typing import ChainMap
import numpy as np
import math
import csv
import random
from numpy.core.arrayprint import printoptions

# calculate the simple matrix multiply, where m1: (n*1) m2: (n*1)
def matrix_multiply(m1, m2):
    # transpose m1 first
    return np.matmul(m1, m2).tolist()

# update W function
def update_W(W, r, gradient_set):
    a1 = matrix_multiply_num(gradient_set, r)
    res = matrix_sub(W, a1)
    return res

# get the value of cost function
def get_cost(X_set, Y_set, W):
    cost = 0
    for (X_i, Y_i) in zip(X_set, Y_set):
        cost += pow_matrix(matrix_sub(Y_i, matrix_multiply(X_i, W)))
    cost /= 2
    return cost

# sum all ele*ele in m up
def pow_matrix(m):
    np_m = np.array(m)
    np_m_1 = np.square(np_m)
    np_m_2 = np.sum(np_m_1)
    return np_m_2

# get the sutract result of two matrix with the same shape
def matrix_sub(m1, m2):
    np_m1 = np.array(m1)
    np_m2 = np.array(m2)
    res = np_m1 - np_m2
    return res.tolist()

# get the addition result of two matrix with the same shape
def matrix_add(m1, m2):
    np_m1 = np.array(m1)
    np_m2 = np.array(m2)
    res = np_m1 + np_m2
    return res.tolist()

# a matrix * number
def matrix_multiply_num(m1, number):
    np_m1 = np.array(m1)
    res = np_m1*number
    return res.tolist()

# implementation of batch gradient descent
def batch_gradient_descent(X_set, Y_set, r, threshould):
    X_length = len(X_set[0])
    Y_length = len(Y_set[0])
    example_num = len(X_set)
    W = []
    norm = 100      # the difference of W
    t = 0           # number of iteration
    cost_set = []       # the value of cost function of each iteration

    # initialize W
    for i in range(X_length):
        W_sub = []
        for j in range(Y_length):
            W_sub.append(0)
        W.append(W_sub)

    while norm > threshould:
        t += 1
        gradient_set = []
        # compute gradient
        for j in range(X_length):
            gradient = []
            for i in range(Y_length):
                gradient.append(0)
            for i in range(example_num):
                a1 = matrix_multiply(X_set[i], W)
                a2 = matrix_sub(Y_set[i], a1)
                a3 = matrix_multiply_num(a2, X_set[i][j])
                gradient = matrix_sub(gradient, a3)
                # print('a1', a1)
                # print('a2', a2)
                # print('a3', a3)
                # print('gradient', gradient)
            gradient_set.append(gradient)
        
        # update W
        W = update_W(W, r, gradient_set)

        # calculate norm
        norm = math.sqrt(pow_matrix(matrix_multiply_num(gradient_set, r)))

        # calculate the value of cost function
        cost = get_cost(X_set, Y_set, W)
        cost_set.append(cost)

        # print
        print('t:', t)
        # print('gradient_set:', gradient_set)
        # print('W:', W)
        print('norm:', norm)
        print('cost:', cost)

    return {"W": W, "t": t, "cost_set": cost_set}

# implementation of stochastic gradient descent
def stochastic_gradient_descent(X_set, Y_set, r, threshould):
    X_length = len(X_set[0])
    Y_length = len(Y_set[0])
    example_num = len(X_set)
    W = []
    cost = 10000000      # the cost function value
    t = 0           # number of iteration
    cost_set = []       # the value of cost function of each iteration

    # initialize W
    for i in range(X_length):
        W_sub = []
        for j in range(Y_length):
            W_sub.append(0)
        W.append(W_sub)

    while cost > threshould:
        t += 1

        # randomly sample a training example
        sample_id = random.randint(0, example_num-1)
        sample_X = X_set[sample_id]
        sample_Y = Y_set[sample_id]

        # update W
        W_new = []
        for j in range(X_length):
            w_t = W[j]
            a_1 = matrix_multiply(sample_X, W)
            a_2 = matrix_sub(sample_Y, a_1)
            a_3 = matrix_multiply_num(a_2, sample_X[j]*r)
            w_t_new = matrix_add(w_t, a_3)
            W_new.append(w_t_new)
        
        # update W
        W = W_new

        # calculate the value of cost function
        cost = get_cost(X_set, Y_set, W)
        cost_set.append(cost)

        # print
        print('t:', t)
        # print('W:', W)
        print('cost:', cost)
    return {"W": W, "t": t, "cost_set": cost_set}


if __name__ == '__main__':
    # process data
    data = []
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []

    with open('./Data/slump_test.data', 'r') as f:
        next(f)
        while True:
            line = f.readline()
            if line:
                line = line.strip('\n').split(',')[1:]
                for i in range(len(line)):
                    line[i] = float(line[i])
                line.insert(0, 1)
                data.append(line)
                train_X.append(line[0:8])
                train_Y.append(line[8:])
                test_X.append(line[0:8])
                test_Y.append(line[8:])
            else:
                break
    train_X = train_X[0:54]
    train_Y = train_Y[0:54]
    test_X = test_X[54:]
    test_Y = test_Y[54:]

    # print(train_X)
    # print(train_Y)
    # train_X = [[1, 1, -1, 2], [1, 1, 1, 3], [1, -1, 1, 0], [1, 1, 2, -4], [1, 3, -1, -1]]
    # train_Y = [[1], [4], [-1], [-2], [0]]
    # test_X = test_X[54:]
    # test_Y = test_Y[54:]

    # para setting
    X_set = train_X      # each X: [1, x_1, x_2, x_3, ...]
    Y_set = train_Y      # each y: [y_1, y_2, y_3, ...]

    #-----------------------------batch gradient descent---------------------------------------------------------
    batch_r = 0.000000025           # learning rate
    batch_threshould = 0.00001    # prove whether converge the difference of W

    batch_res = batch_gradient_descent(X_set, Y_set, batch_r, batch_threshould)
    batch_W = batch_res['W']
    batch_t = batch_res['t']
    batch_cost_set = batch_res['cost_set']
    print('batch_W', batch_W)
    print('batch_t', batch_t)
    print('batch_cost_set', batch_cost_set)

    # restore the cost_set into a file
    new_batch_cost_set = []
    for i in range(batch_t):
        new_batch_cost_set.append([batch_cost_set[i]])
    with open('./Results/4a.csv', 'w') as f:
        csv_f = csv.writer(f, lineterminator='\n')
        csv_f.writerows(new_batch_cost_set)
    print('write the cost function value into 4a.csv')

    # use the learned W to calculate the cost function value of test data
    test_cost = get_cost(test_X, test_Y, batch_W)
    # print('test_cost', test_cost)

    #-----------------------------stochastic gradient descent---------------------------------------------------------
    
    stochastic_r = 0.00000005
    stochastic_threshould = 6600

    stochastic_res = stochastic_gradient_descent(X_set, Y_set, stochastic_r, stochastic_threshould)
    stochastic_W = stochastic_res['W']
    stochastic_t = stochastic_res['t']
    stochastic_cost_set = stochastic_res['cost_set']
    print('stochastic_W', stochastic_W)
    print('stochastic_t', stochastic_t)
    print('stochastic_cost_set', stochastic_cost_set)

    # restore the cost_set into a file
    new_batch_cost_set = []
    for i in range(stochastic_t):
        new_batch_cost_set.append([stochastic_cost_set[i]])
    with open('./Results/4b.csv', 'w') as f:
        csv_f = csv.writer(f, lineterminator='\n')
        csv_f.writerows(new_batch_cost_set)
    print('write the cost function value of stochastic gradient descent into 4b.csv')


    # use the learned W to calculate the cost function value of test data
    test_cost = get_cost(test_X, test_Y, stochastic_W)
    print('test_cost', test_cost)

#-----------------------------4(c)---------------------------------------------------------
# get the X
np_X = np.array(X_set)
np_Y = np.array(Y_set)
W_ = np.dot(np.dot(np.linalg.inv(np.dot(np_X.T, np_X)), np_X.T), np_Y)
# a1 = np.matmul(np.transpose(X_set), X_set)
# a2 = np.linalg.inv(a1)
# a3 = np.matmul(a2, np.transpose(X_set))
# W_ = np.matmul(a3, Y_set)
print('the optimal weight vector')
print(W_)
