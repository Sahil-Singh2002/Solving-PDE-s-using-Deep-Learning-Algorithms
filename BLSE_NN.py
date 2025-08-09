import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def relu(z):
    return np.maximum(0, z)

def basis_function(x, w, b, index):
    if index == 0:
        return 1
    else:
        return np.maximum(0, w * x - b)

def inner_product(w2_i, b2_i, w2_j, b2_j, domain, i, j):
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

def integrate_trapezium(domain, n, func):
    x = np.linspace(domain[0], domain[1], n + 1)
    h = (domain[1] - domain[0]) / n
    weights = h * np.ones(n + 1)
    weights[[0, -1]] = h / 2
    return np.sum(func(x) * weights)

def integrate_romberg(domain, n, func, level):
    R = np.zeros((level + 1, level + 1))
    for i in range(level + 1):
        R[i, 0] = integrate_trapezium(domain, n, func)
        n *= 2
        for j in range(1, i + 1):
            R[i, j] = (R[i, j - 1] * 4**j - R[i - 1, j - 1]) / (4**j - 1)
    return R[level - 1, level - 1]

def neural_network_predict(x, w2, w3, b2, b3):
    z2 = w2 * x + b2
    a2 = relu(z2)
    a3 = w3.T @ a2 + b3
    return a3

def calculate_cost(m_matrix, f_vector, b3, w3, domain, n_points, romberg_level):
    coefficients = np.vstack((b3, w3))
    term1 = float(coefficients.T @ m_matrix @ coefficients)
    term2 = -2 * float(coefficients.T @ f_vector)
    integrand = lambda x: np.cos(np.pi * x)**2
    l2_norm_squared = integrate_romberg(domain, n_points, integrand, romberg_level)
    return term1 + term2 + l2_norm_squared


# --- main parameters ---
DOMAIN = (0, 1)
NUM_DATA_POINTS = 50
X_PLOT = np.linspace(DOMAIN[0], DOMAIN[1], NUM_DATA_POINTS)
Y_PLOT = np.cos(np.pi * X_PLOT)
ROMBERG_LEVEL = 5
TRAPEZIUM_N = NUM_DATA_POINTS - 1

NEURON_SETTINGS = [1, 3, 9]
MARKERS = ['o', 'x', 's']  # per-setting markers

plt.figure(figsize=(10, 6))
tf_handle, = plt.plot(X_PLOT, Y_PLOT, color='black', linewidth=2,
                      label="True function y = cos(πx)")

legend_specs = []  # collect (marker, neurons, color, error)

for marker, num_neurons in zip(MARKERS, NEURON_SETTINGS):
    w2 = np.ones((num_neurons, 1))
    b2 = np.linspace(DOMAIN[0], DOMAIN[1], num_neurons, endpoint=False).reshape(-1, 1)

    m_matrix = np.zeros((num_neurons + 1, num_neurons + 1))
    f_vector = np.zeros((num_neurons + 1, 1))

    for i in range(num_neurons + 1):
        if i == 0:
            f_vector[i] = integrate_romberg(DOMAIN, TRAPEZIUM_N,
                lambda x: np.cos(np.pi * x) * basis_function(x, w2[i], b2[i], i), ROMBERG_LEVEL)
        else:
            f_vector[i] = integrate_romberg(DOMAIN, TRAPEZIUM_N,
                lambda x: np.cos(np.pi * x) * basis_function(x, w2[i-1], b2[i-1], i), ROMBERG_LEVEL)

        for j in range(num_neurons + 1):
            w2_i = w2[i-1] if i > 0 else None
            b2_i = b2[i-1] if i > 0 else None
            w2_j = w2[j-1] if j > 0 else None
            b2_j = b2[j-1] if j > 0 else None
            m_matrix[i, j] = inner_product(w2_i, b2_i, w2_j, b2_j, DOMAIN, i, j)

    # solve system
    L = np.linalg.cholesky(m_matrix)
    y = np.linalg.solve(L, f_vector)
    params = np.linalg.solve(L.T, y)

    b3_opt = params[0:1]
    w3_opt = params[1:]

    # NN curve
    nn_plot = np.array([neural_network_predict(xi, w2, w3_opt, -b2, b3_opt)[0][0]
                        for xi in X_PLOT])
    cost_val = calculate_cost(m_matrix, f_vector, b3_opt, w3_opt,
                              DOMAIN, TRAPEZIUM_N, ROMBERG_LEVEL)

    line, = plt.plot(X_PLOT, nn_plot, linestyle='--')  # no label; we do custom legend
    color = line.get_color()

    # Correct breakpoints: x = b2 / w2 (filter to domain)
    bp_x = (b2.flatten() / w2.flatten())
    bp_x = bp_x[(bp_x >= DOMAIN[0]) & (bp_x <= DOMAIN[1])]
    bp_y = np.array([neural_network_predict(xi, w2, w3_opt, -b2, b3_opt)[0][0] for xi in bp_x])
    plt.scatter(bp_x, bp_y, marker=marker, s=70, color=color)

    legend_specs.append((marker, num_neurons, color, float(cost_val)))

# --- clean combined legend (one entry per neuron count) ---
custom_handles = [Line2D([0], [0], color='black', linewidth=2,
                         label="True function y = cos(πx)")]
for marker, neurons, color, err in legend_specs:
    custom_handles.append(
        Line2D([0], [0], linestyle='--', marker=marker, color=color,
               markersize=6, linewidth=2,
               label=f'{neurons} neurons, Error: {err:.1e}')
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("NN Approximations with ReLU Breakpoints")
plt.legend(
    handles=custom_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.07),
    ncol=2,
    frameon=False,
    fontsize=13, labelspacing=0.6, handlelength=2.2, handletextpad=0.6)
plt.tight_layout()
plt.show()

