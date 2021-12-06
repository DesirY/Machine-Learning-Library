''''
SVM in the primal domain with stochastic sub-gradient descent
'''
import csv
import random
import copy
import math
import re
from time import process_time_ns
from scipy.optimize import minimize


class Dual_SVM:
    def __init__(self, C, training_data, test_data):
        self.C = C
        self.training_data = training_data
        self.test_data = test_data
        self.N = len(training_data)
        self.w = []
        self.b = 0
        
   
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
        print('enter function')
        # calculate the function
        def fun(x_lst, y_lst):
            def f(alpha_lst):
                res = 0
                for i in range(self.N):
                    for j in range(self.N):
                        res += 1/2*(y_lst[i]*y_lst[j]*alpha_lst[i]*alpha_lst[j]*self.matrix_mul(x_lst[i], x_lst[j]))
                for i in range(self.N):
                    res += alpha_lst[i]
                return res   

            return f
        
        # get the constraints
        def con(y_lst):
            cons = ()
            # add inequality constraints
            for i in range(self.N):
                con = ({'type': 'ineq', 'fun': lambda x: x[i]},
                       {'type': 'ineq', 'fun': lambda x: -x[i]+self.C})
                cons += con
            
            # add quality
            def v(alpha_lst):
                res = 0
                for i in range(self.N):
                    res += alpha_lst[i]*y_lst[i]
                return res
            
            cons += ({'type': 'eq', 'fun': v},)
            return cons

        alpha_initial = []
        for i in range(self.N):
            alpha_initial.append(0)

        x_input = []
        y_input = []
        training_set = copy.deepcopy(self.training_data)
        for i in range(self.N):
            x_input.append(training_set[i][0:-1])
            y_input.append(training_set[i][-1])
        
        print('input func')
        input_fun = fun(x_input, y_input)
        print('input func end')
        input_con = con(y_input)
        print('input con end')

        result = minimize(input_fun, alpha_initial, method='SLSQP', constraints=input_con, options={'maxiter': 10})

        alpha_lst = result.x

        print(result.fun)
        print(result.success)
        print(result.x)

        # get w
        self.w = self.get_w(alpha_lst, x_input, y_input)
        print(self.w)
        # get b
        self.b = self.get_b(x_input, y_input)
    
    '''
    get w
    '''
    def get_w(self, alpha_lst, x_lst, y_lst):
        res = [0, 0, 0, 0]
        for i in range(self.N):
            w_1 = self.matrix_mul_num(x_lst[i], alpha_lst[i]*y_lst[i])
            res = self.matrix_add(w_1, res)
        
        return res
    

    def get_b(self, x_lst, y_lst):
        res = 0
        for i in range(self.N):
            cur_res = y_lst[i] - self.matrix_mul(self.w, x_lst[i])
            res += cur_res
        return res/self.N

        
    def test_error(self, test_set):
        test_num = len(test_set)
        test_true_num = 0
        
        for example in test_set:
            example_X = example[:-1]
            example_y = example[-1]

            if (self.matrix_mul(example_X, self.w)+self.b)*example_y > 0:
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

    C = [100/873, 500/873, 700/873]
    
    SVM_solver = Dual_SVM(C[0], training_data, test_data)
    SVM_solver.SVM_processor()
   



    print('----------------------------')
    a = 1
    print('(b) Answer:')
    for i in range(3):
        train_data = copy.deepcopy(training_data)
        SVM_solver = (gama_0, a, C[i], T, train_data, test_data)
        SVM_solver.SVM_processor()
        print('when C = ', C[i])
        print('model parameter:', SVM_solver.w)
        print('training error:', SVM_solver.test_error(training_data))
        print('test error:', SVM_solver.test_error(test_data))
        print('------------------------------------')
