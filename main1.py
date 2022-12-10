from random import seed
from random import random

#Neural network must have 5 input neurons, 4 hidden neurons, and 1 output neuron
#Use the Sigmoid transfer function, 0.2 error threshold and 0.2 learning rate

#Initialize the neural network
def initialize_network(n_inputs, n_hidden, n_outputs): #n_input is the number of input neurons, n_hidden is the number of hidden neurons, n_outputs is the number of output neurons
 #creating a list datastructure which stores the input, hideen and output layer
 network = list() 


 hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]

 #adding the hidden layer to the network
 network.append(hidden_layer)

 output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]

 #adding the output layer to the network
 network.append(output_layer)

 return network

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
 print(layer)

def sigmoid_function():
    return (1/(1 + np.exp(-x)))