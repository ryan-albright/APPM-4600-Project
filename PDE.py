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
        u_0 = 3
        u_n = -2

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

        
        plt.show()
    
    elif dimension == 1:
        n_list = n
        for n in n_list:
            # # Define initial problem # #
        
            # Interval
            a = 0
            b = np.pi

            # Dirichlet Boundary Conditions
            u_0 = 1
            u_n = 1

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
        
        plt.show()
    
    elif dimension == 2:
        # # Define initial problem # #
        
        # Interval
        a, b = -1, 1

        # Dirichlet Boundary Conditions
        gamma_1 = lambda x,y: x*y*0 
        gamma_2 = lambda x,y: x*y*0 - np.sin(x)

        print(f"Here is the numerical solution for 2D Poisson on a square wih Dirichlet boundary conditions u(x,y) = 1 on the boundary")

        # Defining b vector
        x1 = np.linspace(a,b,n+2)
        x2 = np.linspace(a,b,n+2)

        h = (b - a) / (n - 1)

        x1_grid, x2_grid = np.meshgrid(x1,x2)

        # Creating A matrix
        size = n**2
        A1 = np.diag([4]*size)
        A2 = np.diag([-1]*(size-1),1)
        A3 = np.diag([-1]*(size-1),-1)
        A4 = np.diag([-1]*(n**2 - n),n)
        A5 = np.diag([-1]*(n**2 - n),-n)

        A = (-1/h**2)*(A1 + A2 + A3 + A4 + A5)

        for i in range(n**2):
            for j in range(n**2):
                if i % n == 0 and j % n == 0:
                    A[i,j-1] = 0
                    A[i-1,j] = 0

        # print(np.linalg.eigvals(A)) Can run this line to check if SPD or SND...
        # Creating b vector
        b = np.empty((n,n))

        for i in range(1,n+1):
            for j in range(1,n+1):
                xhm, yhm = x1[i-1], x2[i-1]
                x, y = x1[i], x2[j]
                xhp, yhp = x1[i+1], x2[i+1]
                # top left corner
                if i == 1 and j == 1:
                    b[i-1,j-1] = func_2d(x,y) - gamma_1(x,yhp) / h**2 - gamma_2(xhm,y) / h**2
                # top side
                elif j == 1 and i != 1 and i != n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_1(x,yhp) / h**2
                # top right corner
                elif i == n and j == 1:
                    b[i-1,j-1] = func_2d(x,y) - gamma_1(x,yhp) / h**2 - gamma_2(xhp,y) / h**2
                # left side
                elif i == 1 and j != 1 and j != n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_2(xhm,y) / h**2
                # right side
                elif i == n and j != 1 and j != n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_2(xhp,y) / h**2
                # bottom left corner
                elif i == 1 and j == n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_2(xhm,y) / h**2 - gamma_1(x,yhm) / h**2
                # bottom side
                elif j == n and i != 1 and i != n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_1(x,yhm) / h**2
                # bottom right corner
                elif i == n and j == n:
                    b[i-1,j-1] = func_2d(x,y) - gamma_1(x,yhm) / h**2 - gamma_2(xhp,y) / h**2
                # interior points
                else:
                    b[i-1,j-1] = func_2d(x,y)

        b = b.flatten()

        # creating grid of points to plot
        # We basically need to add the boundary points to the outside of the matrix
        bv_mat = np.zeros((n+2, n+2))
                    
        [y, iterates, it_count]  = CG(A,b,10**-8, 5000, True)
                
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        z = y.reshape(n,n)
        # We basically need to add the boundary points to the outside of the matrix
        y1 = gamma_1(x1_grid, x2_grid)
        y2 = gamma_2(x1_grid, x2_grid)

        y1[0,1:n+1] = y2[0,1:n+1]
        y1[n+1,1:n+1] = y2[0,1:n+1]

        y1[1:n+1,1:n+1] = z

        surf = ax.plot_surface(x1_grid, x2_grid, y1, cmap='viridis')
        plt.show()


def func_2d(x, y):
        return -np.sin(x + np.pi/2) - np.cos(y)
        # return np.sin(x)

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

# poisson(1, [10,100,1000])
poisson(2,50)



        

