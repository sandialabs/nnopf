# nnopf
This repository features neural network models implemented in Python, designed to learn solutions for optimal power flow (OPF) problems. 

OPF involves optimizing the delivery of electricity from generating plants to consumers, with the goal of minimizing generation costs while adhering to the physical constraints of the power grid. Because OPF can be computationally challenging to solve using classical optimization approaches, nnopf trains a neural network on diverse data realizations of a user-provided or publicly available power grid exemplar. Once trained, the neural network can predict the optimal solution to the OPF problem for a specified grid exemplar.  
