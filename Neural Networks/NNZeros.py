import numpy as np
import random
import csv
import copy
from numpy.core.arrayprint import printoptions
from numpy.lib import math
import math
from scipy.special import expit, logit


# sigmoid function
def sigmoid(a):
    # Prevent overflow.
    # a = np.clip(a, -500, 500)
    # print(1.0/(1.0+np.exp(-a)))
    # return 1.0/(1.0+np.exp(-a))
    return expit(a)

# the derivative of the sigmoid function
def drv_sigmoid(a):
    return sigmoid(a) * (1-sigmoid(a))

class NNetwork:
    def __init__(self, architecture, gama_0, d, T, training_set, threshold):
        self.architecture = architecture    # [10, 10, 10] means that three layers, with the number of node: 10, 10, 10
        self.layer_num = len(architecture)
        self.gama_0 = gama_0
        self.d = d
        self.T = T
        self.training_set = training_set
        self.B = [np.zeros((l, 1)) for l in architecture[1:]]   # the bias term, the output layer does not have bias term
        # print(self.B)
        for B in self.B:
            B = B.astype(np.float128)
        self.W = [np.zeros((l, next_l)) for l, next_l in zip(architecture[:-1], architecture[1:])]  # normal distribution
        for W in self.W:
            W = W.astype(np.float128)
        self.threshold = threshold
        # print(self.W)

    # x -> input, y->label
    def back_propagation(self, x, y):   
        # init an empty array for the derivative
        drv_B = [np.zeros(b.shape, dtype=np.float128) for b in self.B]
        drv_W = [np.zeros(w.shape, dtype=np.float128) for w in self.W]

        Z = []      # all of the Z
        A = []      # sigmoid Z

        for b, w in zip(self.B, self.W):
            # compute Z for each layer
            z = w.T @ a + b if Z else w.T @ x + b
            a = sigmoid(z)
            Z.append(z)
            A.append(a)

        H = self.layer_num - 2      # the number of the hidden layer
        for L in range(H, -1, -1):
            # from top to bottom to calculate the derivative
            # print(Z[L]-y)
            drv = drv_sigmoid(Z[L]) * (self.W[L+1] @ drv) if L != H else Z[L] - y # the output layer doesn't account for it
            drv_B[L] = drv
            # print(x.T.shape)
            # print(drv.shape)
            drv_W[L] = A[L-1] @ drv.T if L != 0 else x @ drv.T

        return (drv_W, drv_B)
    
    # stochastic gradient descent algorithm
    def SGD(self):
        t = 1
        for epoch in range(self.T):
            # print('epoch', epoch)
            random.shuffle(self.training_set)
            for example in self.training_set:
                example_X = example[0]
                example_Y = example[1]

                gama = self.gama_0/(1+self.gama_0*(t)/self.d)

                t += 1
                
                last_w = copy.deepcopy(self.W)
                last_b = copy.deepcopy(self.B)

                # calculate the gradient of the loss
                partial_W, partial_B = self.back_propagation(example_X, example_Y)

                # Update W and B
                self.W = [W - gama*partial for W, partial in zip(self.W, partial_W)]
                self.B = [B - gama*partial for B, partial in zip(self.B, partial_B)]
                
                # calculate the change of the error
                error = self.cal_error(partial_W, partial_B, gama)
                # print(error)
                if error < self.threshold/(self.architecture[0]*self.architecture[1]+self.architecture[1]*self.architecture[2]+self.architecture[2]*self.architecture[3]):            # 1e-9 效果比较好
                    return


    # cal_error
    def cal_error(self, w, b, gama):
        sum = 0
        for w_ele in w:
            sum += np.sum((w_ele*gama)**2)
        for b_ele in b:
            sum += np.sum((b_ele*gama)**2)

        return math.sqrt(sum)
        

    # predict the result
    def predict(self, x):
        Z = []      # all of the Z
        A = []      # sigmoid Z

        for b, w in zip(self.B, self.W):
            # compute Z for each layer
            z = w.T @ a + b if Z else w.T @ x + b
            a = sigmoid(z)
            Z.append(z)
            A.append(a)
        
        # print('the predict is', Z[-1])
        
        res = 0 if abs(Z[-1]-1) > abs(Z[-1]-0) else 1
        # print(res)

        return res
    
    # test the error rate given examples
    def test_error(self, examples):
        test_num = len(examples)
        test_true_num = 0

        for example in examples:
            example_X = example[0]
            example_y = example[1]
            predict_y = self.predict(example_X)
            
            if example_y == predict_y:
                test_true_num += 1
        
        print(test_true_num/test_num)
        
        return test_true_num/test_num


'''
processing data
[
    [np.array([[x1], [x2]...]), label]
]
'''
def process_data(address):
    result = []
    with open(address, 'r') as f:
        f_csv = csv.reader(f)
        for item in f_csv:
            for i in range(len(item)-1):
                item[i] = float(item[i])
            arr = np.array(item[:-1], dtype=np.float128)
            arr.shape = (4, 1)
            result.append([arr,  int(item[-1])])
            
    return result

if __name__ == '__main__':
    training_data = process_data('./Data/train.csv')
    test_data = process_data('./Data/test.csv')
    # 1, 0.01
    gama_0 = 0.002
    d = 100
    T = 50

    width = [5, 10, 25, 50, 100]
    thresh = [1e-9, 1e-9, 1e-10, 1e-12, 1e-15]
    
    gama_0_lst = [0.002, 0.0023, 0.003, 0.003, 0.003]
    D_lst = [100, 100, 100, 100, 100]

    cnt = 0
    for wid in width:
        # d = D_lst[cnt]
        # gama_0 = gama_0_lst[cnt]
        NN = NNetwork([4, wid, wid, 1], gama_0, d, T, training_data, 1e-3)
        NN.SGD()        # training
        NN.test_error(training_data)
        NN.test_error(test_data)
        cnt += 1
