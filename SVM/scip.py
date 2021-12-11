from scipy.optimize import minimize
import numpy as np
 
def fun(args):
     a=args
     v=lambda x:a/x[0] +x[0]
     return v
 
args = (1)  #a
x0 = [2]  # 初始猜测值
res = minimize(fun(args), x0, method='SLSQP')
print(res.fun)
print(res.success)

