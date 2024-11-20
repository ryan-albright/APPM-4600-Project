import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import norm, qr

def evalF(x):
    
    F = np.zeros(3)
    F[0] = x[0] + math.cos(x[0]*x[1]*x[2]) - 1.
    F[1] = (1.-x[0])**(0.25) + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1
    F[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
    return F

def evalJ(x):
   
    J = np.array([
        [1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]), 
         x[0]*x[2]*math.sin(x[0]*x[1]*x[2]), 
         x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
        [-0.25*(1-x[0])**(-0.75), 1, 0.1*x[2]-0.15],
        [-2*x[0], -0.2*x[1]+0.01, 1]
    ])
    return J

def evalg(x):
   
    F = evalF(x)
    return np.sum(F**2)

def eval_gradg(x):
    
    F = evalF(x)
    J = evalJ(x)
    return J.T @ F

def optimize_step_size(x, z, g1):
    
    alpha3 = 1.0
    dif_vec = x - alpha3*z
    g3 = evalg(dif_vec)
    
    
    while g3 >= g1:
        alpha3 /= 2
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)
    
    alpha2 = alpha3/2
    dif_vec = x - alpha2*z
    g2 = evalg(dif_vec)
    
    
    h1 = (g2 - g1)/alpha2
    h2 = (g3-g2)/(alpha3-alpha2)
    h3 = (h2-h1)/alpha3
    
    alpha0 = 0.5*(alpha2 - h1/h3)
    
    dif_vec = x - alpha0*z
    g0 = evalg(dif_vec)
    
    return (alpha0, g0) if g0 <= g3 else (alpha3, g3)

def linear_step_size(r, A):
    #Main difference between CG and steepest
    return float(r.T @ r / (r.T @ A @ r))


        
    

def solve_linear(A, b, tol=1e-6, Nmax=100):
    
    x = b.copy()
    convergence_history = []
    r = b - A @ x
    z = r.copy()

    r_norm_sq = r.T @ r
    
    for i in range(Nmax):
        Az = A @ z
        alpha =(r_norm_sq / (z.T @ Az)) 
        x_new = x + alpha*z

        #check this in office hours
        r_new = r - alpha * Az

        test = r_new.T @ r
        r_norm_sq_new = r_new.T @ r_new
        
        beta = float(r_norm_sq_new / r_norm_sq)
        
        print (test) # testing if residual is orthogonal
        convergence_history.append(norm(r))
        
        if norm(x_new - x) < tol:
            return x_new, norm(r), 0, "Converged", convergence_history
        x = x_new
        
    return x, norm(r), 1, "Max iterations exceeded", convergence_history

def generate_spd_matrix(n, kappa):
    
    Q, _ = qr(np.random.randn(n, n))
    D = np.diag(np.linspace(1, kappa, n))
    return Q @ D @ Q.T

def driver():
   
    
    plt.figure(figsize=(10, 5))
    for kappa in kappas:
        A = generate_spd_matrix(n, kappa)
        b = np.random.randn(n, 1)
        b = b / norm(b)
        
        x, res, ier, msg, linear_history = solve_linear(A, b)
        
   
    
 

if __name__ == '__main__':
    driver()

'''
Questions for Office Hours
How to find Beta
CG step size differences
'''