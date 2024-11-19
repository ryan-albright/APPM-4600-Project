import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def SD (A, b, tol, nmax):
    '''
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    '''
    # define first guess 
    xk = b

    # iteration
    n = 0
    while n < nmax:
        rk = b - A @ xk
        ak = np.dot(rk, rk) / np.dot(rk, A @ rk)
        xk1 = xk + ak*rk

        if norm(xk1, xk) < tol:
            print('Solution accurate to the given tolerance found')
            return xk1
        xk = xk1
    
    print('Maximum Number of iterations exceeded')


    # if k < nmax:




