import torch
import torch.nn as nn
import numpy as np


def pNetwork(x, y, rule, depth, width):
        myNet = ''
        if(rule == 1):
            myNet = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
            )
            print(myNet)
        else:
            myNet = nn.Sequential(
            nn.Linear(2, 10),
            nn.tanh(),
            nn.Linear(10, 1),
            nn.Sigmoid()
            )
            print(myNet)

        optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
        loss_func = nn.MSELoss()

        for epoch in range(5000):
        out = myNet(x)
        loss = loss_func(out, y)  
        optimzer.zero_grad()  
        loss.backward()
        optimzer.step()

        print(myNet(x).data)

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
   
    depth = [3, 5, 9]
    width = [5, 10, 25, 50, 100]
    
    print('ReLU'):
    for wid in width:
        for dep in depth:
            pNetwork(training_data, 1, wid, dep, 0)
            test_error(test_error)
    print('tanh'):
    for wid in width:
        for dep in depth:
            pNetwork(training_data, 1, wid, dep, 0)
            test_error(test_error)