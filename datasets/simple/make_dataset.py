import numpy as np
import pickle
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem

from scipy import io as sio
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

seed = 17
num_var = 100
num_ineq = 50
num_eq = 50
num_examples = 3600


num_ineq = num_ineq
for num_eq in [num_eq]:
    print(num_ineq, num_eq)
    np.random.seed(seed)
    P = np.array([np.diag(np.random.random(num_var)) for _ in range(num_examples)])
    c = np.random.random((num_examples ,num_var))
    A = np.random.normal(loc=0, scale=1., size=(num_examples, num_eq, num_var))
    b = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_examples, num_ineq, num_var))
    h = np.array([np.sum(np.abs(G[i]@np.linalg.pinv(A[i])), axis=1) for i in range(num_examples)])

    l = -10 * np.ones((num_examples, num_var))
    u = 10 * np.ones((num_examples, num_var))
    
    data = {'P': P, 'c': c, 'A': A, 'b': b, 'G': G, 'h': h, 'l': l, 'u': u}
    
    # save the data into a mat file
    # with open('datasets/simple/simple_problem_{}_{}_{}_{}.mat'.format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
    #     sio.savemat(f, data)
    
    
    problem = SimpleProblem(data, valid_num=300, test_num=300, calc_X=True)

    with open("/datasets/simple/random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples), 'wb') as f:
        pickle.dump(problem, f)