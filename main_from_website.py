import pandas
import numpy as np

def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

#error threshold
error_threshold = 0.2

#learning rate
LR = 0.2

#input neurons layer
I_dim = 3

#hidden neurons layer
H_dim = 4

#epoch counter
epoch_count = 1000

np.random.seed(1) #to always have the same random weights

#an array of arrays, each input neuron has an array of weights 
# of which size is equal to the number of hidden neurons
weights_ItoH = np.random.uniform(-1, 1 , (I_dim, H_dim))

#an array for the hidden to output weights, equal to the number of hidden neurons
weights_HtoO = np.random.uniform(-1, 1 , H_dim)

#an array to store the hidden neurons before the activation function (sigmoid)
preActivation_H = np.zeros(H_dim)

#an array to store the hidden neurons after the activation function (sigmoid)
postActivation_H = np.zeros(H_dim)

#importing the training data
training_dataset = pandas.read_csv('training_dataset.csv')

#the target output, selecting the 'Survived' column
target_output = training_dataset.Survived

#removing the survived column from the dataset
training_dataset = training_dataset.drop(['Survived'], axis=1)

#converting the dataset into an array of arrays
training_dataset = np.asarray(training_dataset)

#store the number of training samples, i.e the number of records in the dataset
training_count = len(training_dataset[:,0])

#Feed forward processing

#process will keep on repeating depending on the 'epoch_count' value
for epoch in range(epoch_count):

    #bad facts
    bad_facts = 0

    #good facts
    good_facts = 1

    #loop used to each time select a different record of values from the dataset
    for sample in range(training_count):
        #loop to point to the different hidden neurons, node is referring to hidden neuron
        for node in range(H_dim):
            # calculating the pre activation value of the current hidden neuron, np.dot is the product of two values, 
            # in this case multiplying two arrays, one of input neurons and 
            # the other of the respective weights pointing to the current hidden neuron
            preActivation_H[node] = np.dot(training_dataset[sample,:], weights_ItoH[:, node])
            
            #using the activation function to calculate the value of the hidden neuron
            postActivation_H[node] = logistic(preActivation_H[node])
        
        #after calculating the value of each hidden neuron, we calculate the output neuron by first multiplying the hidden neurons with the respective weights,
        # here it is being assumed that there is only one neuron in the output neuron layer
        preActivation_O = np.dot(postActivation_H, weights_HtoO)

        #calculating the final value of the output neuron
        postActivation_O = logistic(preActivation_O)

        #final error, i.e. error margin, the achieved value of the output neuron subtracted by the target output of that record of values
        FE = postActivation_O - target_output[sample]

        if(FE<0):
            FE = FE * -1
        
        if(FE > error_threshold):
            #if the error margin is greater than the error threshold we increase the bad facts counter
            bad_facts = bad_facts + 1

            #error back propagation incase FE is larger than error threshold

            #create an array to store the delta/error of the output neurons
            delta_o = np.zeros(1)

            #create an array to store the delta/error of the output neurons
            delta_h = np.zeros(H_dim)

            #calculate error for the output neurons first
            delta_o[0] = (postActivation_O[0] - target_output[sample]) * logistic_deriv(postActivation_O[0])

            #calculate the error for the hidden neurons
            
        else:
            #if the error margin is less than the error threshold we increase the good facts counter
            good_facts = good_facts + 1

