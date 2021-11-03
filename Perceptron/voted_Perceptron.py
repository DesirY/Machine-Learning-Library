import csv
import random
from typing import Iterator

class Voted_Perceptron:
    training_set = []   # by default, we thinik this the last element is Y value
    T = ''
    w = []
    num = 0
    r = ''
    w_c_set = []
    m = 0

    def __init__(self, training_set, T, r):
        self.training_set = self.augment_training_set(training_set)
        self.T = T
        self.num = len(training_set)
        self.r = r
        for i in range(len(training_set[0])-1):
            self.w.append(0)
    
    '''
    augment the first element as 1
    '''
    def augment_training_set(self, training_set):
        res = []
        for example in training_set:
            example.insert(0, 1)
            res.append(example)
        return res

    '''
    A and B is one dimension data
    '''
    def matrix_mul(self, A, B):
        res = 0
        for a, b in zip(A, B):
            res += a*b
        return res
    
    '''
     A is one dimension data, A * num
    '''
    def matrix_mul_num(self, A, num):
        res = []
        for a in A:
            res.append(a*num)
        return res

    '''
    A and B is one dimension data
    '''
    def matrix_add(self, A, B):
        res = []
        for a, b in zip(A, B):
            res.append(a+b)
        return res
    
    def voted_perceptron(self):
        c = ''
        for epoch in range(self.T):
            random.shuffle(self.training_set)
            for example in self.training_set:
                example_X = example[:-1]
                example_y = example[-1]
               
                judge = self.matrix_mul(self.w, example_X)*example_y      #judge the prediction
                if judge <= 0:      # update the w
                    if self.m != 0:
                        self.w_c_set.append([self.w, c])
                    update_w = self.matrix_mul_num(example_X, self.r*example_y)
                    self.w = self.matrix_add(self.w, update_w)
                    self.m += 1
                    c = 1
                else:
                    c += 1
        self.w_c_set.append([self.w, c])

    def test_error(self, test_set):
        test_set = self.augment_training_set(test_set)
        test_num = len(test_set)
        test_true_num = 0

        for example in test_set:
            example_X = example[:-1]
            example_y = example[-1]
            predict = 0
            
            for w_c in self.w_c_set:
                w = w_c[0]
                c = w_c[1]
                if self.matrix_mul(example_X, w) > 0:
                    predict += 1*c
                else:
                    predict -= 1*c
            
            if predict*example_y > 0:
                test_true_num += 1
        
        return 1 - test_true_num/test_num


'''
processing data
'''
def process_data(address):
    result = []
    with open(address, 'r') as f:
        f_csv = csv.reader(f)
        for item in f_csv:
            res = item
            for i in range(len(res)):
                res[i] = float(res[i])
            if res[-1] > 0.5:
                res[-1] = 1
            else:
                res[-1] = -1
            result.append(res)
    return result


if __name__ == '__main__':
    training_data = process_data('./Data/train.csv')
    test_data = process_data('./Data/test.csv')
    T = 10
    r = 0.3
    perceptron = Voted_Perceptron(training_data, T, r)
    perceptron.voted_perceptron()
    print('Voted Perceptron')
    print('----------------------------------------------------------')
    print('Weight Vectors', 'Count')
    for W_C in perceptron.w_c_set:
        print(W_C[0], W_C[1])
    print('test error', perceptron.test_error(test_data))
