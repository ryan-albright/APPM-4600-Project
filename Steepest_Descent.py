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

    # Small Test
    # A = np.array([[2,1],[1,2]])
    # b = np.array([[4],[3]])

    [x, iterates] = SD(A, b, 1e-6, 200)
    

def SD (A, b, tol, nmax):
    '''
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    Outputs:
    xk - steepest descent solution
    its - the iterations taken to find the solution
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
            print('Solution accurate to the given tolerance found')
            print(f'Algorithm took {i} iterations to find the solution') 
            return [xk1, iterates[:i+1,:]] # trims off zeros
        xk = xk1

    print('Maximum Number of iterations exceeded')
    return [0, iterates[:i+1,:]] 

driver(100,30)