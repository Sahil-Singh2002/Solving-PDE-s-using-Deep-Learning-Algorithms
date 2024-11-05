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
    if index == 0:  # Handle case for i,j = 0
        return 1  # Return 1 for the constant basis function
    else:
        return np.maximum(0, w * x + b)# in the paper ReLU(wi *x - bi) for the basis function i,j = 1,...,n

print("np.maximum(0, w * x + b)")

# Define the inner product of two basis functions over the domain Omega
def inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j): # Added i, j arguments
    # Define the integrand as the product of two ReLU-activated basis functions
    integrand = lambda x: phi(x, W2_i, b2_i, i) * phi(x, W2_j, b2_j, j) # Pass i, j to phi
    
    # Compute the integral over the specified domain
    result, _ = quad(integrand, domain[0], domain[1])
    return result
#________________________________________
#Numerical Computation for the projected
def composite_trapezium(domain,n,f):
    """
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the composite trapezium rule with n subintervals.   
    Returns
    -------
    integral_approx (float): The approximation of the integral 
    """
    
    x = np.linspace(domain[0],domain[1],n+1) #Construct the quadrature points
    h = (domain[1]-domain[0])/n

    #Construct the quadrature weights: 
    #These are the coefficient w_i of f(x_i) in the summation
    weights = h*np.ones(n+1) 
    weights[[0,-1]] = h/2

    integral_approx = np.sum(f(x)*weights)

    return integral_approx

def romberg_integration(domain,n,f,level):
    '''
    To approximate the definite integral of the function f(x) over the interval [a,b]
    using the Richardson extrapolation to CTR(n) with the specified level of extrapolation (level).

    Returns
    -------
    integral_approx (float): This gives the approximated integral of the
    function f(x) in the interval [a,b] for Romberg integration approximation based on
    CTR(level*n). Giving R_(level,level).
    '''
    R = np.zeros((level+1,level+1))
    for i in range(0,level+1): 
        R[i,0]=composite_trapezium(domain,n,f)
        n = 2*n
        for j in range(1,i+1):
            R[i,j] = (R[i,j-1]*4**(j) -R[i-1,j-1])/(4**(j) -1)

    integral_approx = R[level-1,level-1]
    return integral_approx
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
    costval = np.abs((b - a) / Number_of_data_point) * np.linalg.norm(costvec) ** 2  
    return costval
#________________________________________
# Input for the NN
Number_of_data_point = 25  # Increase to 25-50 with respect to the domain.
Number_of_Neurons_in_Hidden_Layer = 3  # The greater the number of neurons, the higher the accuracy, but with a tradeoff in time complexity.
print(f"Number of Neurons in Hidden layer = {Number_of_Neurons_in_Hidden_Layer}" )
# X is the collection of x_i I want to work with, which must be uniformly distributed
a, b = 0, 1  # Domain size
domain = (a, b) 
X = np.random.uniform(a, b, Number_of_data_point) # points need to be random but follows the uniform dist.
y = np.cos(np.pi * X)

# Initialize Inner Parameters
W2 = np.ones((Number_of_Neurons_in_Hidden_Layer, 1))
b2 = -1* np.linspace(a,b,Number_of_Neurons_in_Hidden_Layer, endpoint=False).reshape(-1,1) #this should be linspace where the points should be uniformely spaced
# this was resulting in the randomness of the NN output. 

print("Initialized W2:", W2)
print("Initialized b2:", b2)
# ________________________________________
# Least-Squares Initialization for Outer Parameters
# Compute phi_matrix (ReLU activations with an extra column for the constant term i,j = 0)
M = np.zeros((Number_of_Neurons_in_Hidden_Layer+1, Number_of_Neurons_in_Hidden_Layer+1))
F = np.zeros((Number_of_Neurons_in_Hidden_Layer+1,1 ))
# Compuational parameters for the Romberg integration
level = 5
n = Number_of_data_point - 1

# Populate the matrix M with the inner products and the projection vector F using the numerical computation
for i in range(Number_of_Neurons_in_Hidden_Layer+1):
    if i == 0 : # Compute F[0] for the constant basis function (index = 0)
        F[i] = romberg_integration(domain, n, lambda x: np.cos(np.pi * x) * phi(x, W2[i], b2[i], i), level)
    else :
        F[i] = romberg_integration(domain, n, lambda x: np.cos(np.pi * x) * phi(x, W2[i-1], b2[i-1], i), level)
    for j in range(Number_of_Neurons_in_Hidden_Layer+1):
        # Use conditional indexing for W2 and b2
        W2_i = W2[i-1] if i > 0 else None
        b2_i = b2[i-1] if i > 0 else None
        W2_j = W2[j-1] if j > 0 else None
        b2_j = b2[j-1] if j > 0 else None

        M[i, j] = inner_product(W2_i, b2_i, W2_j, b2_j, domain, i, j) # Pass i, j

# Display the resulting matrix
print("Matrix M:")
print(M) # this checks out with the hand written calculations
# Display the resulting vector
print("Vector F:")
print(F) # this checks out with the hand written calculations

# Solve for W3 and b3
params = np.linalg.solve(M,F)

b3 = params[0:1]  # Initial bias for output layer
W3 = params[1:]  # Initial weights for output layer

print("Initialized W3:", W3)
print("Initialized b3:", b3)

#________________________________________
#Initial plot of the NN with the Best Least Squares where now the W2 = [1,...,1] and b2 = [0,..., 1] the initial parameters are now set with n number of 1's for W2 and a linspace of b2 with number of elements beign the number of neurons used in hidden layer.
# The outer parameter is just solving for Mc = F where c = [b3, <- W3 -> ] and now plotting the initual plot for NN uisng the best parameters.

x_plot = np.linspace(a, b, Number_of_data_point)
y_plot = np.cos(np.pi *x_plot)
NN_plot = np.array([neural_network(xi, W2, W3, b2, b3)[0][0] for xi in x_plot])
COST = cost(W2, W3, b2, b3)

plt.plot(x_plot, y_plot, label="True function y = cos(pix)", color='blue')
plt.plot(x_plot, NN_plot, label="Initual Neural Network Approx.", color='red', linestyle='--')
plt.scatter(X, y, color='green', label="Data points")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Initual Neural Network Approximation\nCost: {COST:.6f}")
plt.legend(loc="best")
plt.show()

#________________________________________
# Training Loop
eta = 0.05
Epoch =  (10**4) 
costs = np.zeros(Epoch)

for i in tqdm(range(Epoch), desc="Training Progress"):
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
    b3 = b3 - eta * delta3 # get ride of this when null phi is ignored

    W2 = W2 - eta * delta2 * x
    W3 = W3 - eta * delta3 * a2

    # Compute and store cost
    newcost = cost(W2, W3, b2, b3) 
    costs[i] = (newcost)

#________________________________________
# Plotting the Cost Reduction Over Iterations
ite = [ i for i in range(Epoch)]
plt.plot(ite, np.log(costs))
plt.xlabel("Epoch")
plt.ylabel("Log(Cost)")
plt.title("Log(Cost) Reduction Over Epoch")
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
