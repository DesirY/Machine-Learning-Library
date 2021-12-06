''''
SVM in the primal domain with stochastic sub-gradient descent
'''
import csv
import random
import copy
import math

class Sto_Sub_Gradient_SVM:
    def __init__(self, gama_0, a, C, T, training_data, test_data):
        self.gama_0 = gama_0
        self.a = a
        self.C = C
        self.T = T
        self.training_set = self.augment_training_set(training_data)
        self.test_data = test_data
        self.N = len(training_data)
        self.w = []
        for i in range(len(training_data[0])):
            self.w.append(0)
        self.cnverge_rate = 3e-6

    '''
    augment the first element as 1
    '''
    def augment_training_set(self, training_set):
        train_data = copy.deepcopy(training_set)
        res = []
        for example in train_data:
            example.insert(4, 1)
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
    
    '''
    A and B is one dimension data
    '''
    def matrix_sub(self, A, B):
        res = []
        for a, b in zip(A, B):
            res.append(a-b)
        return res
    
    '''
    calculate error
    '''
    def cal_error(self, A, B):
        res = 0
        for a, b in zip(A, B):
            res += pow(a-b, 2)
        return math.sqrt(res)

    def SVM_processor(self):
        t = 1
        for epoch in range(self.T):
            random.shuffle(self.training_set)
            for example in self.training_set:
                example_X = list(example[:-1])
                example_y = example[-1]
            
                judge = self.matrix_mul(self.w, example_X)*example_y      #judge the prediction
                #print('judge', judge)

                gama = self.gama_0/(1+self.gama_0*t/self.a)
                #print('gama', gama)

                t += 1
                
                last_w = copy.deepcopy(self.w)

                if judge <= 1:      # update the w
                    w_full = list(self.w[:-1])
                    w_full.append(0)
                    #print('w_full', w_full)
                    w_1 = self.matrix_mul_num(w_full, gama)
                    #print('w_1', w_1)
                    w_2 = self.matrix_mul_num(example_X, gama*self.C*self.N*example_y)
                    #print('w_2', w_2)
                    self.w = self.matrix_add(self.matrix_sub(self.w, w_1), w_2)

                else:
                    w_1 = self.matrix_mul_num(list(self.w[:-1]), 1-gama)
                    for index, value in enumerate(self.w):
                        if index < 4:
                            self.w[index] = w_1[index]
                
                error = self.cal_error(self.w, last_w)
                # print('t', t, 'w:', self.w)
                # print(error)
                if error < self.cnverge_rate:
                    #print('converge')
                    return


    def test_error(self, test_set):
        test_set = self.augment_training_set(test_set)
        test_num = len(test_set)
        test_true_num = 0
        
        for example in test_set:
            example_X = example[:-1]
            example_y = example[-1]

            if self.matrix_mul(example_X, self.w)*example_y > 0:
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

    gama_0 = 1
    a = 0.5
    C = [100/873, 500/873, 700/873]
    T = 900
    
    print('(a) Answer:')
    for i in range(3):
        train_data = copy.deepcopy(training_data)
        SVM_solver = Sto_Sub_Gradient_SVM(gama_0, a, C[i], T, train_data, test_data)
        SVM_solver.SVM_processor()
        print('when C = ', C[i])
        print('model parameter:', SVM_solver.w)
        print('training error:', SVM_solver.test_error(training_data))
        print('test error:', SVM_solver.test_error(test_data))
        print('------------------------------------')

    print('----------------------------')
    a = 1
    print('(b) Answer:')
    for i in range(3):
        train_data = copy.deepcopy(training_data)
        SVM_solver = Sto_Sub_Gradient_SVM(gama_0, a, C[i], T, train_data, test_data)
        SVM_solver.SVM_processor()
        print('when C = ', C[i])
        print('model parameter:', SVM_solver.w)
        print('training error:', SVM_solver.test_error(training_data))
        print('test error:', SVM_solver.test_error(test_data))
        print('------------------------------------')
