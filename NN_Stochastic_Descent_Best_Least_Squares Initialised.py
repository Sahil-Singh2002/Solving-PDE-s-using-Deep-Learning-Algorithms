#________________________________________
#Library
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from tqdm import tqdm

#________________________________________
#Sigmoid activation
def activation(z):
    return np.where(z>=0,z,0) #this is for ReLU
    #return 1 / (1 + np.exp(-z)) # This is for Sigmoid

def derivative(z):
    return np.where(z>0,1,0) #this is for ReLU 
    #return activation(z) * (1 - activation(z)) # This is for Sigmoid
#________________________________________
# Define the basis function with parameters (weights and bias)
def phi(x, w, b, index): # Added 'index' argument
    if index == 0:
        return 1  # Return 1 for the basis function i,j = 0
    else:
        return max(0, (np.dot(w, x) + b) )# in the paper ReLU(wi *x - bi) for the basis function i,j = 1,...,n

# Define the inner product of two basis functions over the domain Omega
def inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j): # Added i, j arguments
    # Define the integrand as the product of two ReLU-activated basis functions
    integrand = lambda x: phi(x, W2_i, b2_i, i) * phi(x, W2_j, b2_j, j) # Pass i, j to phi
    
    # Compute the integral over the specified domain
    result, _ = quad(integrand, domain[0], domain[1])
    return result
#________________________________________
# Define Artificial Neural Network
def neural_network(x, W2, W3, b2, b3):
    z2 = W2 * x + b2 # should be W2.T x + b2 once we are dealing with PDE
    a2 = activation(z2)
    a3 = W3.T @ a2 + b3 # No activation function implemented as this is a regularization problem, not classification.
    return a3
#________________________________________
# Define the Stochastic Descent Method
def cost(W2, W3, b2, b3):
    costvec = np.zeros(Number_of_data_point)
    for j in range(Number_of_data_point):
        x = X[j]
        a3 = neural_network(x, W2, W3, b2, b3)  
        costvec[j] = y[j] - a3[0][0]
    costval = (abs(b - a) / Number_of_data_point) * np.linalg.norm(costvec) ** 2  
    return costval
#________________________________________
# Input for the NN
Number_of_data_point = 25  # Increase to 25-50 with respect to the domain.
Number_of_Neurons_in_Hidden_Layer = 5  # The greater the number of neurons, the higher the accuracy, but with a tradeoff in time complexity.
# X is the collection of x_i I want to work with, which must be uniformly distributed
a, b = 0, 1  # Domain size
domain = (a, b) 
X = np.random.uniform(a, b, Number_of_data_point) #
y = np.cos(np.pi * X)

# Initialize Inner Parameters
W2 = np.ones((Number_of_Neurons_in_Hidden_Layer, 1))
b2 = np.linspace(a,b,Number_of_Neurons_in_Hidden_Layer).reshape(-1,1) #this should be linspace where the points should be uniformely spaced
# this was resulting in the randomness of the NN output. 


print("Initialized W2:", W2)
print("Initialized b2:", b2)
# ________________________________________
# Least-Squares Initialization for Outer Parameters
# Compute phi_matrix (ReLU activations with an extra column for the constant term i,j = 0)
M = np.zeros((Number_of_Neurons_in_Hidden_Layer+1, Number_of_Neurons_in_Hidden_Layer+1))
# Populate the matrix M with the inner products
for i in range(n+1):
    for j in range(n+1):
        # Use conditional indexing for W2 and b2
        W2_i = W2[i-1] if i > 0 else None
        b2_i = b2[i-1] if i > 0 else None
        W2_j = W2[j-1] if j > 0 else None
        b2_j = b2[j-1] if j > 0 else None

        M[i, j] = inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j) # Pass i, j

# Display the resulting matrix
print("Matrix M:")
print(M) # this checks out with the hand written calculations





# Solve for W3 and b3
params = np.linalg.solve(M, F)

b3 = params[0:1]  # Initial bias for output layer
W3 = params[1:]  # Initial weights for output layer

print("Initialized W3:", W3)
print("Initialized b3:", b3)


x_plot = np.linspace(a, b, Number_of_data_point)
y_plot = np.cos(np.pi *x_plot)
NN_plot = np.array([neural_network(xi, W2, W3, b2, b3)[0][0] for xi in x_plot])
COST = cost(W2, W3, b2, b3)

plt.plot(x_plot, y_plot, label="True function y = cos(pix)", color='blue')
plt.plot(x_plot, NN_plot, label="Neural network approximation initual", color='red', linestyle='--')
plt.scatter(X, y, color='green', label="Data points")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Comparison of True Function and Neural Network Approximation\nCost: {COST:.6f}")
plt.legend(loc="best")
plt.show()

#________________________________________
# Training Loop
eta = 0.05
Niter =  (1) 
costs = np.zeros(Niter)

for i in tqdm(range(Niter), desc="Training Progress"):
    k = np.random.randint(0, Number_of_data_point)
    x = X[k]

    # Forward pass
    z2 = W2 * x + b2
    a2 = activation(z2)
    z3 = W3.T @ a2 + b3
    a3 = z3

    # Backpropagation
    delta3 = -2 *(y[k] - a3[0][0])
    delta2 = derivative(z2) * (W3) * delta3

    # Update weights and biases
    b2 = b2 - eta * delta2
    b3 = b3 - eta * delta3

    W2 = W2 - eta * delta2 * x
    W3 = W3 - eta * delta3 * a2

    # Compute and store cost
    newcost = cost(W2, W3, b2, b3)
    costs[i] = newcost

#________________________________________
# Plotting the Cost Reduction Over Iterations
plt.semilogy(range(Niter), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.show()

#________________________________________
# Plot the Actual Function and the Neural Network Output
x_plot = np.linspace(a, b, Number_of_data_point)
y_plot = np.cos(np.pi *x_plot)
NN_plot = np.array([neural_network(xi, W2, W3, b2, b3)[0][0] for xi in x_plot])
COST = cost(W2, W3, b2, b3)

plt.plot(x_plot, y_plot, label="True function y = cos(pix)", color='blue')
plt.plot(x_plot, NN_plot, label="Neural network approximation", color='red', linestyle='--')
plt.scatter(X, y, color='green', label="Data points")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Comparison of True Function and Neural Network Approximation\nCost: {COST:.6f}")
plt.legend(loc="best")
plt.show()
