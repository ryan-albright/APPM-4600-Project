import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time


def driver (n,K):
    # create a symmetric positive definite matrix
    [Q, R] = np.linalg.qr(np.random.randn(n,n))
    D = np.diag(np.linspace(1,K,n))
    
    A = Q @ D @ Q.T
    b = np.random.randn(n,1)
    b = b / norm(b)

    # Small Test
    # A = np.array([[2,1],[1,2]])
    # b = np.array([[4],[3]])
    print(np.linalg.cond(A), D)
    [x, iterates, it_count] = SD(A, b, 1e-8, 5000)

    # Convergence Experiments
    print(f'Please see the below experiment to test convergence with n = {n} and kappa = {K}')
    diffs = iterates - x.T
    err = np.array([norm(diff) for diff in diffs])
    rate = np.average(err[1:] / err[:-1])
    plt.semilogy(err)
    plt.ylabel('en+1 - en')
    plt.xlabel('Iteration')
    plt.show()
    print(f'The order of convergence is linear with rate {rate}')

    # Time experiments
    print(f'Please see below the experiments performed to test run time with n = {n} and kappa = {K}')
    t_0 = time.time()
    for j in range(100):
        [x, iterates, it_count] = SD(A, b, 1e-8, 5000, verb = False)
    t_1 = time.time()
    print(f'Average time: {(t_1 - t_0)/100} seconds')
    print(f'Number of iterations: {it_count}')

def SD (A, b, tol, nmax, verb = True):
    '''
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    Outputs:
    xk - steepest descent solution
    its - the iterates calculated to find the solution
    i - the number of iterations taken
    '''
    # define first guess
    iterates = np.zeros([nmax, b.size])
    
    x = b.copy()
    convergence_history = []
    r = b - A @ x
    z = r.copy()
    r_norm_sq = r.T @ r

    
    
    # iteration
    for i in range(nmax):
        Az = A @ z
        alpha =(r_norm_sq / (z.T @ Az))
        x_new = x + alpha * z
        r_new = x + alpha * z
        r_nrom_sq_new = r_new.T @ r_new
        beta = float(r_nrom_sq_new / r_norm_sq)


        

        if norm(x_new - x) < tol: # is this the correct way to check tolerance or is it w ak?
            if verb:
                print('Solution accurate to the given tolerance found')
                print(f'Algorithm took {i} iterations to find the solution') 
            return [x_new, iterates[:i,:], i] # trims off zeros
        
    if verb:
        print('Maximum Number of iterations exceeded')
    return [0, iterates[:i,:], i] 

driver(100,100000)