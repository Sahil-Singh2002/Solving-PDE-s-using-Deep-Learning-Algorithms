import numpy as np
import matplotlib.pyplot as plt

# Activation (ReLU)
def activation(z):
    return np.where(z >= 0, z, 0)

# Basis function
def phi(x, w, b, index):
    if index == 0:
        return 1
    else:
        return np.maximum(0, w * x - b)

# Inner product without scipy
def inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j):
    if i == 0 and j == 0:
        return domain[1] - domain[0]
    elif i == j:
        return (domain[1] - b2_i)**3 / 3
    elif j == 0 and i != 0:
        return (domain[1] - b2_i)**2 / 2
    elif i == 0 and j != 0:
        return (domain[1] - b2_j)**2 / 2
    elif i > j:
        return (1/3)*(domain[1]**3 - b2_i**3) - (1/2)*(domain[1]**2 - b2_i**2)*(b2_i + b2_j) + b2_i*b2_j*(domain[1] - b2_i)
    elif j > i:
        return (1/3)*(domain[1]**3 - b2_j**3) - (1/2)*(domain[1]**2 - b2_j**2)*(b2_i + b2_j) + b2_i*b2_j*(domain[1] - b2_j)

# Composite trapezium
def composite_trapezium(domain, n, f):
    x = np.linspace(domain[0], domain[1], n + 1)
    h = (domain[1] - domain[0]) / n
    weights = h * np.ones(n + 1)
    weights[[0, -1]] = h / 2
    return np.sum(f(x) * weights)

# Romberg integration
def romberg_integration(domain, n, f, level):
    R = np.zeros((level + 1, level + 1))
    for i in range(level + 1):
        R[i, 0] = composite_trapezium(domain, n, f)
        n *= 2
        for j in range(1, i + 1):
            R[i, j] = (R[i, j - 1] * 4**j - R[i - 1, j - 1]) / (4**j - 1)
    return R[level - 1, level - 1]

# Neural network
def neural_network(x, W2, W3, b2, b3):
    z2 = W2 * x + b2
    a2 = activation(z2)
    a3 = W3.T @ a2 + b3
    return a3

# Cost function
def compute_cost(M, F, b3, W3, domain, n, level):
    c = np.vstack((b3, W3))
    term1 = float(c.T @ M @ c)
    term2 = -2 * float(c.T @ F)
    integrand = lambda x: np.cos(np.pi * x)**2
    l2_norm_squared = romberg_integration(domain, n, integrand, level)
    return term1 + term2 + l2_norm_squared

# Main parameters
domain = (0, 1)
Number_of_data_point = 50
x_plot = np.linspace(domain[0], domain[1], Number_of_data_point)
y_plot = np.cos(np.pi * x_plot)
level = 5
n = Number_of_data_point - 1

neuron_settings = [3, 5, 8, 10]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, num_neurons in enumerate(neuron_settings):
    W2 = np.ones((num_neurons, 1))
    b2 = np.linspace(domain[0], domain[1], num_neurons, endpoint=False).reshape(-1, 1)

    print(f"Number of neurons: {num_neurons}")
    print(f"W2: {W2}")
    print(f"b2: {b2}")

    M = np.zeros((num_neurons + 1, num_neurons + 1))
    F = np.zeros((num_neurons + 1, 1))

    for i in range(num_neurons + 1):
        if i == 0:
            F[i] = romberg_integration(domain, n, lambda x: np.cos(np.pi * x) * phi(x, W2[i], b2[i], i), level)
        else:
            F[i] = romberg_integration(domain, n, lambda x: np.cos(np.pi * x) * phi(x, W2[i - 1], b2[i - 1], i), level)

        for j in range(num_neurons + 1):
            W2_i = W2[i - 1] if i > 0 else None
            b2_i = b2[i - 1] if i > 0 else None
            W2_j = W2[j - 1] if j > 0 else None
            b2_j = b2[j - 1] if j > 0 else None
            M[i, j] = inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j)

    L = np.linalg.cholesky(M)
    y = np.linalg.solve(L, F)        # Solve L y = F
    params = np.linalg.solve(L.T, y) # Solve Lᵀ x = y ⇒ x = params


    print(f"M: {M}")
    print(f"F: {F}")
    print(f"Params: {params}")

    b3 = params[0:1]
    W3 = params[1:]

    NN_plot = np.array([neural_network(xi, W2, W3, -b2, b3)[0][0] for xi in x_plot])
    cost_val = compute_cost(M, F, b3, W3, domain, n, level)

    ax = axes[idx]
    ax.plot(x_plot, y_plot, label="True function y = cos(πx)", color='blue')
    ax.plot(x_plot, NN_plot, label="NN Approximation", color='red', linestyle='--')
    ax.set_title(f"{num_neurons} Neurons — Cost: {cost_val:.6f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

plt.tight_layout()
plt.show()
