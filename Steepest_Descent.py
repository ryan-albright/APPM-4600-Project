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


    x = SD(A, b, 1e-6, 50)


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

    # iteration
    n = 0
    while n < nmax:
        rk = b - A @ xk
        ak = np.inner(rk, rk) / np.inner(rk, A @ rk)
        xk1 = xk + ak*rk

        x = norm(xk1, xk)
        print(xk1, xk)

        if norm(xk1, xk) < tol:
            print('Solution accurate to the given tolerance found')
            return xk1
        n += 1
        xk = xk1

    print('Maximum Number of iterations exceeded')
    return 0 




