#________________________________________
#Library
import numpy as np
import matplotlib.pyplot as plt 
#________________________________________
#Sigmoid activation
def activation(z):
    #return np.where(z>=0,z,0) #this is for ReLU
    return 1 / (1 + np.exp(-z)) # This is for Sigmoid

def derivative(z):
    #return np.where(z>0,1,0) #this is for ReLU 
    return activation(z) * (1 - activation(z)) # This is for Sigmoid
#________________________________________
# Define Artifical Neural Network
def neural_network(x, W2, W3, b2, b3):
    z2 = W2 * x + b2 # should be W2.T x + b2 once we are dealing with with PDE
    a2 = activation(z2)
    a3 = W3.T @ a2 + b3 ## No activation function implemented as this is a regularisation problem and not Classification problem.
    return a3
#________________________________________
# Define the Stochastic descent Method
def cost(W2, W3, b2, b3):
    costvec = np.zeros(Number_of_data_point)
    for j in range(Number_of_data_point):
        x = X[j]
        
        a3 = neural_network(x, W2, W3, b2, b3)  
        
        costvec[j] = y[j] - a3[0][0]
    costval = (abs(b-a)/Number_of_data_point) * np.linalg.norm(costvec) **2  
    return costval
#________________________________________
# Input for the NN
Number_of_data_point = 100 ## Increase 25 - 50 respect to the domain.
Number_of_Neurons_in_Hidden_Layer = 5 ## The greater the number of neurons in the hidden layer the greater the accuracy however you trade in the time space complexity O()
#X is the collection of x_i i want to work with where it must be uniformaly distributed
a, b = -2*np.pi, 2*np.pi # Domain size
X = np.random.uniform(a, b, Number_of_data_point) 
y = np.cos(X)

# Randomly initializing weights and biases
W2 , W3 = 0.5*np.random.normal(size=(Number_of_Neurons_in_Hidden_Layer,1)) , 0.5*np.random.normal(size=(Number_of_Neurons_in_Hidden_Layer,1))
b2 , b3 = 0.5*np.random.normal(size=(Number_of_Neurons_in_Hidden_Layer,1)) , 0.5*np.random.normal(size=(1,1))

eta = 0.05
Niter = 10**4

costs = np.zeros(Niter)
#________________________________________
# Training loop
for i in range(Niter):
    k = np.random.randint(0,Number_of_data_point) 
    # Choose at random which data point to train NN on on each iteration.    
    x = X[k]  

    # Forward pass
    z2 = W2 * x + b2 # should be W2.T x + b2 once we are dealing with with PDE
    a2 = activation(z2)
    
    z3 = W3.T @a2 + b3
    a3 = z3
    
    # Backpropagation
    delta3 = -2*(y[k]- a3[0][0]) # normally derivative(z3) o (a3 - y[k]) which is the Hadamard product when dealing with PDE it would be very important to consider this
    delta2 =  derivative(z2) * (W3) * delta3 # Hadamard product a * b for matrix use a@b
    
    # Update weights and biases
    b2 = b2 - eta * delta2
    b3 = b3 - eta * delta3

    W2 = W2 - eta * delta2 *  x # once dealing with PDE we will have x be x.T
    W3 = W3 - eta * delta3 * a2 # once dealing with PDE we will have as be a2.T
    
    # Compute and store cost
    newcost = cost(W2, W3, b2, b3)
    costs[i] = newcost

#________________________________________
# Plotting the cost reduction over iterations
plt.semilogy(range(Niter), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Iterations")
plt.show()
#________________________________________
# Plot the actual function and the neural network output /// New in introduction to see performance for 3 layer NN
x_plot = np.linspace(a, b, Number_of_data_point)
y_plot = np.cos(x_plot)
NN_plot = np.array([neural_network(xi, W2, W3, b2, b3)[0][0] for xi in x_plot])
COST = cost(W2, W3, b2, b3)


plt.plot(x_plot , y_plot, label="True function y = cos(pi * x)", color='blue')
plt.plot(x_plot, NN_plot , label="Neural network approximation", color='red', linestyle='--')
plt.scatter(X, y, color='green', label="Data points")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Comparison of True Function and Neural Network Approximation\nCost: {COST:.6f}")
plt.legend(loc = "best")
plt.show()
