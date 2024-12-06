import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from mpl_toolkits import mplot3d
import time

# Questions:
# How do I implement other boundary conditions?
# 


# 1D Poisson Equation (-u''(x) = f(x))
def poisson(dimension, n):
    if dimension == 1:
        print("Here is the numerical solution for -u''(x) = cos(x) wih Dirichlet boundary conditions u(x_0) = u(x_n) = 0")
        # Using Dirichlet Boundary Conditions u(x_0) = u(x_n) = 0

        # Define b vector (values of f)
        x = np.linspace(0,1,n)
        h = x[1] - x[0]

        f = lambda x: np.cos(x)

        b = f(x)

        # Creating A matrix
        A1 = np.diag([2]*n)
        A2 = np.diag([-1]*(n-1), 1)
        A3 = np.diag([-1]*(n-1),-1)

        A = (1/h**2)*(A1 + A2 + A3)

        [y, iterates, it_count]  = CG(A,b,10**-8, 3000, True)

        # Plot of the solution to the PDE
        plt.plot(x,y)
        plt.title(f'Plot of u(x) solving the 1D Possions equation for f(x) = cos(x)')
        plt.show()
    elif dimension == 2:
        print("Here is the numerical solution for 2D Poisson on a rectange wih Dirichlet boundary conditions u(x) = 0 on the boundary")

        # Defining b vector
        a = -5
        b = 5
        x1 = np.linspace(a,b,n)
        x2 = np.linspace(a,b,n)

        h = (b - a) / (n + 1)

        g1, g2 = np.meshgrid(x1,x2)
        
        f = func_2d(g1,g2)
        
        b = f.flatten()

        # Creating A matrix
        size = n**2
        A1 = np.diag([4]*size)
        A2 = np.diag([-1]*(size-1),1)
        A3 = np.diag([-1]*(size-1),-1)
        A4 = np.diag([-1]*(n**2 - n),n)
        A5 = np.diag([-1]*(n**2 - n),-n)

        A = (1/h**2)*(A1 + A2 + A3 +A4 + A5)

        [y, iterates, it_count]  = CG(A,b,10**-8, 3000, True)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        z = y.reshape(n,n)
        surf = ax.plot_surface(g1, g2, z, cmap='viridis')

        plt.show()

def func_2d(x, y):
        return np.sin(x) + np.cos(y)

def CG(A, b, tol, nmax, verb=True):
    '''
    Conjugate Gradient Method
    Inputs: 
    A - SPD matrix of size n x n  
    b - matrix of size n x 1
    tol - required tolerance
    nmax - maximum number of iterations
    verb - verbose output flag
    Outputs:
    x - solution
    iterates - the iterates calculated to find the solution
    i - the number of iterations taken
    '''
    # Get true solution for error computation
    # x_star = np.linalg.solve(A, b)
    
    # Initialize
    x = np.zeros_like(b)  # Start from zero initial guess
    iterates = np.zeros([nmax, b.size])
    iterates[0,:] = x.flatten()
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
        
        # Check convergence
        if norm(r_new) < tol:
            if verb:
                print(f'Converged in {i+1} iterations')
            return x_new, iterates[:i+1, :], i+1#, x_star
        
        # Compute beta for conjugate direction
        beta = float((r_new.T @ r_new) / (r.T @ r))
        
        # Update search direction
        p = r_new + beta * p
        
        # Prepare for next iteration
        x = x_new
        r = r_new

    # If maximum iterations reached
    if verb:
        print('Maximum Number of iterations exceeded')
    return x, iterates[:i+1, :], i+1#, x_star        


poisson(2, 30)
        

