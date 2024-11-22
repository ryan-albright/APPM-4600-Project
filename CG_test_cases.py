import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time

def SD(A, b, tol, nmax):
    '''
    Steepest Descent Method
    '''
    # Initialize
    iterates = np.zeros([nmax, b.size])
    
    x = np.zeros_like(b)  # Start from zero initial guess
    r = b - A @ x  # Initial residual
    
    # Iteration
    for i in range(nmax):
        # Store current iterate
        iterates[i, :] = x.flatten()
        
        # Compute step length
        Ap = A @ r
        alpha = float((r.T @ r) / (r.T @ Ap))
        
        # Update solution
        x = x + alpha * r
        r = r - alpha * Ap
        
        # Check convergence
        if norm(r) < tol:
            return x, iterates[:i+1, :], i+1
    
    return x, iterates, nmax

# Set up the plot for convergence
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

# Parameters
n = 100
k_values = [10**2, 10**4, 10**6, 10**8, 10**10, 10**12]

# Tracking time experiments
time_results = []
iteration_results = []

# Calculate and plot for different k values
for k in k_values:
    # Convergence plot
    [Q, R] = np.linalg.qr(np.random.randn(n,n))
    D = np.diag(np.linspace(1,k,n))
    
    A = Q @ D @ Q.T
    b = np.random.randn(n,1)
    b = b / norm(b)

    print(f'Experiments for n = {n} and kappa = {k}')
    
    # Time experiments
    t_0 = time.time()
    total_iterations = 0
    for j in range(100):
        [x, iterates, it_count] = SD(A, b, 1e-8, 10000)
        total_iterations += it_count
    
    t_1 = time.time()
    avg_time = (t_1 - t_0)/100
    avg_iterations = total_iterations/100
    
    print(f'Average time: {avg_time} seconds')
    print(f'Average iterations: {avg_iterations}')
    
    # Store for later plotting
    time_results.append(avg_time)
    iteration_results.append(avg_iterations)
    
    # Convergence plot for this k
    diffs = iterates - x.T
    err = np.array([norm(diff) for diff in diffs])
    plt.semilogy(err, label=f'κ = {k}')

plt.title('Convergence Rates for Different Condition Numbers')
plt.ylabel('||en+1 - en||')
plt.xlabel('Iteration')
plt.legend()
plt.grid(True)

# Time and Iterations plot
plt.subplot(1, 2, 2)
plt.plot(k_values, time_results, marker='o', label='Average Time')
plt.plot(k_values, iteration_results, marker='s', label='Average Iterations')
plt.title('Time and Iterations vs Condition Number')
plt.xlabel('Condition Number (κ)')
plt.ylabel('Seconds / Iterations')
plt.xscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print results for reference
print("\nTime Results:", time_results)
print("Iteration Results:", iteration_results)