import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def driver (n,K):
    # create a symmetric positive definite matrix
    [Q, R] = np.linalg.qr(np.random.randn(n,n))
    D = np.diag(np.linspace(1,K,n))
    
    A = Q @ D @ Q.T
    b = np.random.randn(n,1)
    b = b / norm(b)

    A = np.array([[2,1],[1,2]])
    b = np.array([[4],[3]])
    x = SD(A, b, 1e-6, 40)


def SD (A, b, tol, nmax):
    '''
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    Outputs:
    xk - steepest descent solution
    '''
    # define first guess 
    xk = b
    its = np.empty([nmax, b.size])
    its[0,:] = xk.T
    
    # iteration
    n = 0
    for i in range(1,nmax):
        rk = b - A @ xk
        ak = np.dot(rk.T, rk) / np.dot(rk.T, A @ rk)
        xk1 = xk + ak*rk
        its[i] = xk1.T

        if norm(xk1 - xk) < tol: 
            print('Solution accurate to the given tolerance found')
            return xk1
        n += 1
        xk = xk1

    print('Maximum Number of iterations exceeded')
    return 0 

driver(10,1)