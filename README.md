# Solving PDE's using Deep-Learning Algorithms

Partial Differential Equations (PDEs) in second-dimensional domains are common in fields like Fluid/Thermo-Dynamics. Traditional methods for solving these PDEs are computationally expensive. Recently, Deep Neural Networks (DNNs) have been identified as a promising alternative for solving high-dimensional PDEs efficiently. This project focuses on studying and implementing deep-learning algorithms for numerically approximating second-dimensional variational PDEs, and by implementing a non quadrature approach for approximating the minimisation of the energy function.
 
For more details, you can refer to the [Springer article](https://link.springer.com/article/10.1007/s40304-018-0127-z).


We begin with a differential equation:
$\ -\Delta u(x) = f(x), \quad \forall x \in \Omega, \quad u(x) = 0 \quad \forall x \in \partial \Omega \$

PINNs, also known as Deep Least Squares (Deep LSQ), leverage the physical laws expressed as Partial Differential Equations to inform the training process of a neural network, in this case the the Neural Network is trained to satisfy the differential equation and the boundary condion. The objective wouuld be to minimuse the loss function of both the PDE and boundary/initual conditon combined. 
