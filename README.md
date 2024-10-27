# Solving PDE's using Deep-Learning Algorithms

Partial Differential Equations (PDEs) in high-dimensional domains are common in fields like finance, quantum mechanics, and uncertainty quantification. Traditional methods for solving these PDEs are computationally expensive. Recently, Deep Neural Networks (DNNs) have been identified as a promising alternative for solving high-dimensional PDEs efficiently. This project focuses on studying and implementing deep-learning algorithms for numerically approximating high-dimensional variational PDEs.
 
For more details, you can refer to the [Springer article](https://link.springer.com/article/10.1007/s40304-018-0127-z).


We begin with a differential equation:
$\ u'(x) = f(x), \quad \forall x \in \Omega := (0,1), \quad u(0) = 0. \$

PINNs, also known as Deep Least Squares (Deep LSQ), leverage the physical laws expressed as Partial Differential Equations to inform the training process of a neural network, in this case the the Neural Network is trained to satisfy the differential equation and the boundary condion.

the objective wouuld be to minimuse the loss function of both the PDE and boundary/initual conditon combined. 

. The objective cucrently is to approximate the function $\  u(x)  \$ using a neural network:
$\ \ u(x) \approx \text{NN}(x) := u_{\phi}(x), \ \$ where $\ phi \$ represents the parameters (weights and biases) of the network.
