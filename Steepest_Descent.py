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

    [x, iterates, it_count] = SD(A, b, 1e-8, 5000)

    # Order of Convergence Experiments
    print(f'Please see the experiment for order of convergence with n = {n} and kappa = {K}')
    diffs = iterates - x.T
    err = [norm(diff) for diff in diffs]
    plt.semilogy(err)
    plt.ylabel('en+1 - en')
    plt.xlabel('Iteration')
    plt.show()

    # Time experiments
    print(f'Please see below the experiments performed to test run time')
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
    xk = b
    iterates = np.zeros([nmax, b.size])
    iterates[0,:] = xk.T
    
    # iteration
    for i in range(1,nmax):
        rk = b - A @ xk
        ak = np.dot(rk.T, rk) / np.dot(rk.T, A @ rk)
        xk1 = xk + ak*rk
        iterates[i,:] = xk1.T

        if norm(xk1 - xk) < tol: # is this the correct way to check tolerance or is it w ak?
            if verb:
                print('Solution accurate to the given tolerance found')
                print(f'Algorithm took {i} iterations to find the solution') 
            return [xk1, iterates[:i,:], i] # trims off zeros
        xk = xk1
    if verb:
        print('Maximum Number of iterations exceeded')
    return [0, iterates[:i,:], i] 

driver(10,19)