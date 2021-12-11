import numpy as np
from numpy.core.arrayprint import printoptions


# sigmoid function
def sigmoid(a):
    return 1.0/(1.0+np.exp(-a))

# the derivative of the sigmoid function
def drv_sigmoid(a):
    return sigmoid(a) * (1-sigmoid(a))

# x -> input, y->label
# W: the weight martrix 
# B: the weight matrix related to bias
# layer_num: the number of layer
def back_propagation(x, y, W, B, layer_num):   
    # init an empty array for the derivative
    drv_B = [np.zeros(b.shape) for b in B]
    drv_W = [np.zeros(w.shape) for w in W]

    Z = []      # all of the Z
    A = []      # sigmoid Z

    for b, w in zip(B, W):
        # compute Z for each layer
        z = w.T @ a + b if Z else w.T @ x + b
        a = sigmoid(z)
        Z.append(z)
        A.append(a)

    print(Z)
    print(A)

    H = layer_num - 2      # the number of the hidden layer
    for L in range(H, -1, -1):
        # from top to bottom to calculate the derivative
        # print(Z[L]-y)
        drv = drv_sigmoid(Z[L]) * (W[L+1] @ drv) if L != H else Z[L] - y # the output layer doesn't account for it
        drv_B [L] = drv
        if L == 0:
            print(drv.shape)
            print(x.shape)

        drv_W [L] = A[L-1] @ drv.T if L != 0 else x @ drv.T
        
    return (drv_B, drv_W)

if __name__ == '__main__':
    # init weight
    W = [np.array([[-2, 2], [-3, 3]]), np.array([[-2, 2], [-3, 3]]), np.array([[2], [-1.5]])]
    B = [np.array([[-1], [1]]), np.array([[-1], [1]]), np.array([-1])]

    res_B, res_W = back_propagation(np.array([[1], [1]]), 1, W, B, 4)

    print('-------------------Debug Back-Propagation algorithm with paper problem 3------------------')
    cnt = 1
    for w, b in zip(res_W, res_B):
        print('---------------------------------------------')
        print('The result of the ' + str(cnt) +'\'st layer:')
        print('Weights related to the bias term')
        print(b)
        print('Other Weights of this layer')
        print(w)
        cnt += 1

