import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from mpl_toolkits import mplot3d
import pandas as pd


# 1D Poisson Equation (-u''(x) = f(x))
def poisson(dimension, n):
    if dimension == 1 and isinstance(n,int):
        # # Define initial problem # #
        
        # Interval
        a = 0
        b = np.pi

        # Dirichlet Boundary Conditions
        u_0 = 5
        u_n = 2

        # u''(x) = f(x)
        f = lambda x: -np.cos(x) # Change print statement and graph title when you change this

        print(f"Here is the numerical solution for -u''(x) = -cos(x) wih Dirichlet boundary conditions u({a}) = {u_0} and u({b}) = {u_n}")

        # Define b vector (values of f)
        x = np.linspace(a,b,n)
        h = x[1] - x[0]

        b = f(x)

        # Adding Dirichlet Conditions to the b vector
        b[0] = b[0] - u_0 / h**2
        b[-1] = b[-1] - u_n / h**2

        # Creating A matrix
        A1 = np.diag([2]*n)
        A2 = np.diag([-1]*(n-1), 1)
        A3 = np.diag([-1]*(n-1),-1)

        A = (-1/h**2)*(A1 + A2 + A3)

        [y, iterates, it_count]  = CG(A,b,10**-8, 3000, True)

        # Plot of the numerical solution to the PDE
        plt.plot(x,y)
        plt.title(f'Plot of u(x) solving the 1D Possions equation for f(x) = -cos(x)')

        # Plot of the analytical solution to the PDE
        # x1 = np.linspace(a,b)
        # plt.plot(x, np.cos(x))

        # plt.legend(['Numerical', 'Actual'])

        plt.show()
    
    elif dimension == 2:
        # # Define initial problem # #
        
        # Interval
        a = -np.pi / np.sqrt(2)
        b = np.pi / np.sqrt(2)

        # Dirichlet Boundary Conditions
        j = 0

        print("Here is the numerical solution for 2D Poisson on a square wih Dirichlet boundary conditions u(x,y) = {j} on the boundary")

        # Defining b vector
        x1 = np.linspace(a,b,n)
        x2 = np.linspace(a,b,n)

        h = (b - a) / (n + 1)

        g1, g2 = np.meshgrid(x1,x2)

        # Coordinate Rotation
        alpha = 1 / np.sqrt(2)
        
        g1_new = g1 * alpha - g2 * alpha
        g2_new = g1 * alpha + g2 * alpha
        
        f = func_2d(g1_new,g2_new)
        
        b = f.flatten()

        # Adding Dirichlet Conditions to the b vector

        # Make first n+1 entries zero
        b[:n+1] = np.array([j]*(n+1))

        # Make middle entries 0 where needed
        for i in range(n**2):
            if i % n == 0:
                b[i-1] = j
                b[i] = j

        # Make last n+1 entries zero
        b[-n:] = np.array([j]*(n))
                
        # Creating A matrix
        size = n**2
        A1 = np.diag([4]*size)
        A2 = np.diag([-1]*(size-1),1)
        A3 = np.diag([-1]*(size-1),-1)
        A4 = np.diag([-1]*(n**2 - n),n)
        A5 = np.diag([-1]*(n**2 - n),-n)

        # A = (-1/h**2)

        A = (-1/h**2)*(A1 + A2 + A3 +A4 + A5)

        # Modifying top and bottom of A for boundary conditions
        A[:n+1,:] = np.hstack((np.eye(n+1), np.zeros((n + 1, n**2 - n - 1))))

        A[-n:,:] = np.hstack(( np.zeros((n, n**2 - n)),np.eye(n)))

        for i in range(n+1,n**2):
            if i % n == 0:
                row_i = np.zeros(n**2)
                row_i[i] = 1
                A[i,:] = row_i
                row_i1 = np.zeros(n**2)
                row_i1[i-1] = 1
                A[i-1,:] = row_i1
        
        [y, iterates, it_count]  = CG(A,b,10**-8, 5000, True)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        z = y.reshape(n,n)
        surf = ax.plot_surface(g1, g2, z, cmap='viridis')

        plt.show()

def func_2d(x, y):
        return -np.sin(x + np.pi/2) - np.cos(y)

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

poisson(1, 1000)


        

