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

    

def SD(A, b, tol, nmax, verb=True):
    '''
    Conjugate Gradient Method
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    Outputs:
    xk - solution
    its - the iterates calculated to find the solution
    i - the number of iterations taken
    '''
    # Initialize
    iterates = np.zeros([nmax, b.size])
    
    x = np.zeros_like(b)  # Start from zero initial guess
    r = b - A @ x  # Initial residual
    p = r.copy()   # Initial search direction
    
    # Iteration
    for i in range(nmax):
        # Store current iterate
        iterates[i, :] = x.flatten()
        
        # Compute step length
        Ap = A @ p
        alpha = float((r.T @ r) / (p.T @ Ap))
        
        # Update solution and residual
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
               
        # Compute beta (conjugacy parameter)
        beta = float((r_new.T @ r_new) / (r.T @ r))
        
        # Update search direction
        p = r_new + beta * p
        
        # Prepare for next iteration
        x = x_new
        r = r_new

    # If maximum iterations reached
    if verb:
        print('Maximum Number of iterations exceeded')
    return [x, iterates[:nmax, :], nmax]


# Run the driver
driver(100, 100000)
