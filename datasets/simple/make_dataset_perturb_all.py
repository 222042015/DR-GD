import numpy as np
import pickle
import gzip
import torch

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], os.pardir, os.pardir))
from utils import SimpleProblem

from scipy import io as sio
from scipy.sparse import csc_matrix
import scs
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.set_default_dtype(torch.float64)

seed = 17
num_var = 500
num_ineq = num_var // 2
num_eq = num_var // 2
num_examples_train = 1200
num_examples_test = 0
valid_num = 0
test_num = 0
factor = 0.1
density = 0.5 # for 1000 and 2000

# define the save_dir
save_dir = "/datasets/simple"
folder_name_train = "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples_train)
folder_name_test = "random_simple_dataset_var{}_ineq{}_eq{}_ex{}".format(num_var, num_ineq, num_eq, num_examples_test)

if not os.path.exists(os.path.join(save_dir, folder_name_train)):
    os.makedirs(os.path.join(save_dir, folder_name_train))

if not os.path.exists(os.path.join(save_dir, folder_name_test)):
    os.makedirs(os.path.join(save_dir, folder_name_test))

def perturb(P, c, A, b, G, h):
    #perturb quadratic matrix
    chosed_var = np.arange(num_var)
    chosed_ineq = np.arange(num_ineq)
    chosed_eq = np.arange(num_eq)
    
    ini_q_matrix = P.copy()
    ini_p_vec = c.copy()
    ini_ineq_matrix = G.copy()
    ini_ineq_rhs = h.copy()
    ini_eq_matrix = A.copy()
    ini_eq_rhs = b.copy()
    
    q_matrix = ini_q_matrix[chosed_var, :]
    q_matrix = q_matrix[:, chosed_var]
    nonzero_indices = np.nonzero(q_matrix)
    nonzero_values = q_matrix[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    q_matrix[nonzero_indices] = perturbed_values

    #perturb coefficient vector
    p_vec = ini_p_vec.copy()
    nonzero_indices = np.nonzero(ini_p_vec)
    nonzero_values = ini_p_vec[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    p_vec[nonzero_indices] = perturbed_values
    p_vec = p_vec[chosed_var]

    #perturb coefficient vector
    ineq_matrix = ini_ineq_matrix.copy()
    nonzero_indices = np.nonzero(ini_ineq_matrix)
    nonzero_values = ini_ineq_matrix[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    ineq_matrix[nonzero_indices] = perturbed_values
    ineq_matrix = ineq_matrix[chosed_ineq, :]
    ineq_matrix = ineq_matrix[:, chosed_var]

    #perturb coefficient vector
    ineq_rhs = ini_ineq_rhs.copy()
    nonzero_indices = np.nonzero(ini_ineq_rhs)
    nonzero_values = ini_ineq_rhs[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    ineq_rhs[nonzero_indices] = perturbed_values
    ineq_rhs = ineq_rhs[chosed_ineq]

    #perturb coefficient vector
    eq_matrix = ini_eq_matrix.copy()
    nonzero_indices = np.nonzero(ini_eq_matrix)
    nonzero_values = ini_eq_matrix[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    eq_matrix[nonzero_indices] = perturbed_values
    eq_matrix = eq_matrix[chosed_eq, :]
    eq_matrix = eq_matrix[:, chosed_var]

    #perturb coefficient vector
    eq_rhs = ini_eq_rhs.copy()
    nonzero_indices = np.nonzero(ini_eq_rhs)
    nonzero_values = ini_eq_rhs[nonzero_indices]
    perturbed_values = nonzero_values * (1 + factor * np.random.uniform(-1, 1, size=nonzero_values.shape))
    eq_rhs[nonzero_indices] = perturbed_values
    eq_rhs = eq_rhs[chosed_eq]
    
    return q_matrix, p_vec, eq_matrix, eq_rhs, ineq_matrix, ineq_rhs


def solve_scs(P, c, A, b, G, h, lb=None, ub=None, verbose=False, tol=1e-4):
    if A.shape[0] == 0 and b.shape[0] == 0:
        my_A = np.zeros((1, P.shape[0]))
        my_b = np.zeros((1, ))
    else:
        my_A = A
        my_b = b
        
    if G.shape[0] and h.shape[0]:
        my_A = np.vstack([my_A, G])
        my_b = np.hstack([my_b, h])
    
    if lb is not None:
        lb_idx = np.arange(len(lb))[lb != -np.inf]
        if len(lb_idx) > 0:
            A_lb = -np.eye(P.shape[0])[lb_idx, :]
            b_lb = -lb[lb_idx]
            my_A = np.vstack([my_A, A_lb])
            my_b = np.hstack([my_b, b_lb])
    
    if ub is not None:
        ub_idx = np.arange(P.shape[0])[ub != np.inf]
        if len(ub_idx) > 0:
            A_ub = np.eye(P.shape[0])[ub_idx, :]
            b_ub = ub[ub_idx]
            my_A = np.vstack([my_A, A_ub])
            my_b = np.hstack([my_b, b_ub])
    
    if A.shape[0] == 0 and b.shape[0] == 0:
        my_A = my_A[1:]
        my_b = my_b[1:]
        
    cone_dict = {'z': A.shape[0], 'l': my_A.shape[0]-A.shape[0]}
    data = {'P': csc_matrix(P), 'c': c, 'A': csc_matrix(my_A), 'b': my_b, 'cone': cone_dict}
    # solver = scs.SCS(data, cone_dict, eps_abs=tol, eps_rel=tol, verbose=verbose, 
    #                 acceleration_lookback=0, normalize=False, adaptive_scale=False,
    #                 rho_x=1.0, scale=1.0, alpha=1.0)
    solver = scs.SCS(data, cone_dict, verbose=True, max_iters=10000, use_indirect=False)
    
    sol = solver.solve()
    print(sol['info']['pobj'])
    
    if sol['info']['status'] == "solved":
        return sol['x'], sol['y'], sol['s'], sol['info']['iter'], sol['info']['pobj']
    else:
        return None, None, None, None, None



print(num_var, num_ineq, num_eq)
np.random.seed(seed)
P_ini = np.diag(np.random.random(num_var))

# indices1 = np.random.choice(num_var, int(num_var*0.4), replace=False) 
# indices2 = np.random.choice(num_var, int(num_var*0.4), replace=False)
# P_ini[indices1, indices2] = np.random.random(int(num_var*0.4)) * 0.5
# P_ini[indices2, indices1] = P_ini[indices1, indices2]


c_ini = np.random.random(num_var)
A_ini = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
mask = np.random.rand(num_eq, num_var) < density
A_ini = A_ini * mask
b_ini = np.random.uniform(-1, 1, size=(num_eq))
G_ini = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
mask = np.random.rand(num_ineq, num_var) < density
G_ini = G_ini * mask
h_ini = np.sum(np.abs(G_ini@np.linalg.pinv(A_ini)), axis=1)

P = []
c = []
A = []
b = []
G = []
h = []
# l = -np.ones(num_var) * 5
# u = np.ones(num_var) * 5
l = None
u = None

if l is not None and np.all(l == -np.inf):
    l = None
if u is not None and np.all(u == np.inf):
    u = None

X = []
Y = []
S = []
ITER = []
OBJ_VAL = []

count = 0
while count < num_examples_train:
    PP, cc, AA, bb, GG, hh = perturb(P_ini, c_ini, A_ini, b_ini, G_ini, h_ini)
    x, y, s, iter, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u, verbose=True)
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iter, 'obj': obj}

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_train, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
    print(count)

count = 0
while count < num_examples_test:
    PP, cc, AA, bb, GG, hh = perturb(P_ini, c_ini, A_ini, b_ini, G_ini, h_ini)
    x, y, s, iter, obj = solve_scs(PP, cc, AA, bb, GG, hh, l, u, verbose=True)
    if x is not None:
        data = {'P': PP, 'c': cc, 'A': AA, 'b': bb, 'G': GG, 'h': hh, 'l': l, 'u': u,
                'X': x, 'Y': y, 'S': s, 'iter': iter, 'obj': obj}

        # save the data as .gz file
        with gzip.open(os.path.join(save_dir, folder_name_test, "instance_{}.gz".format(count)), 'wb') as f:
            pickle.dump(data, f)
        count += 1
    print(count)