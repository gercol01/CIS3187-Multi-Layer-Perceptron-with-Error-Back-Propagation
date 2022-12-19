import pandas #used to import the training data
import numpy as np #used for matrix operations
import matplotlib.pyplot as plt #used to plot the graph
from sklearn import preprocessing #used for initial normalisation of the dataset

#our transfer / activation function is sigmoid
def sigmoid_function(x):
    return (1/(1 + np.exp(-x)))

#delta function for the output neurons
def delta_function_output(output, target):
    return (output) * (1 - output) * (target - output)

#delta function used for the hidden neurons, the derivative with respect to weight
def delta_function_hidden(output, weight, delta):
    return (output) * (1- output) * (weight) * (delta)

#weights function used during error back propagation
def weights_function(learning_rate, delta, output):
    return (learning_rate) * (delta) * (output)

#let us import the dataset

#importing the training data
training_dataset = pandas.read_csv('training_dataset.csv')

#the target output for the training data
target_output = training_dataset.Survived.values

#we need to correct the fare column, some numbers have two points in the column
fare_array = training_dataset.Fare.values

#iterating through each value in the fare column
for x in fare_array:
    dot_counter = 0
    original_value = x

    for character in x:
        if character =='.':
            dot_counter = dot_counter + 1

    if (dot_counter>1):
        #in the case there are more than 1 '.'
        x = x.replace('.','',1)

        training_dataset['Fare'] = training_dataset['Fare'].replace({original_value: x})

#the survived column is not part of the input data, therefore we remove it
training_data_without_survived = training_dataset.drop(['Survived'], axis=1)

#normalising the data
scaler = preprocessing.MinMaxScaler()
names = training_data_without_survived.columns
d = scaler.fit_transform(training_data_without_survived)
normalised_training_data = pandas.DataFrame(d, columns=names)
normalised_training_data.head()

#convert the training data into a matrix (array of arrays) with the use of numpy
normalised_matrix = np.asarray(normalised_training_data)

#number of input neurons in the input layer
input_layer = 5

#number of hidden neurons in the hidden layer
hidden_layer = 4

#number of output neurons in the output layer
output_layer = 1

#only 1 bias neuron
bias = 1

#instantiating the input to hidden layer neurons
weights_input_to_hidden = np.random.uniform(-1, 1, (hidden_layer, input_layer)) #4 columns, 5 rows

#hidden to output layer neurons
weights_hidden_to_output = np.random.uniform(-1, 1, (output_layer, hidden_layer)) #1 column, 4 rows

#bias weights input to hidden
bias_input_to_hidden = np.zeros((hidden_layer, bias)) #4 columns, 1 row

#bias weights hidden to output
bias_hidden_to_output = np.zeros((output_layer, bias)) #1 column, 1 row

#instantiating the learning rate
learn_rate = 0.2

#error threshold, changed to 0.5 to improve accuracy
error_threshold = 0.5

#the number of epochs
epochs = 1000

#graph variables

#array to store the epoch number in the current iteration
epoch_array = []

#array to store the bad facts in the current iteration
bad_facts_array = []

print("--- Testing dataset results: ---")

#repeat until the epoch counter reaches the 'epochs' value
for epoch in range(epochs):
    #good facts counter
    good_facts = 0

    #bad facts counter
    bad_facts = 0

    #iterating through the records of the dataset
    for record in range(len(normalised_matrix)):
        #input values for the current record number
        input_values = normalised_matrix[record]

        #changing the array into a vertical matrix of arrays which are all size 1
        input_values = np.reshape(input_values, (len(input_values), 1))

        #getting the target value for the current record
        target_value = target_output[record]

        # multiply the weights with the input data, afterwards add the bias weights, 
        # the result is before the summation, 4 arrays for each hidden neuron

        #multiply weights with input values
        product_input = np.dot(weights_input_to_hidden, input_values)

        hidden_neurons_pre = bias_input_to_hidden + (product_input)

        # calculate the values of the hidden neurons using the activation function
        hidden_neurons = sigmoid_function(hidden_neurons_pre) #hidden neuron values

        # multiply the weights with the hidden neurons, afterwards add the bias weights, 
        # the result is before the summation, 1 arrays for each hidden neuron

        #multiply weights with input values
        product_hidden = np.dot(weights_hidden_to_output, hidden_neurons)

        output_neurons_pre = bias_hidden_to_output + (product_hidden) # multiply the weights with the hidden neurons, afterwards add the bias weights

        #calculate the values of the output neurons using the activation function
        output_neurons = sigmoid_function(output_neurons_pre)

        # Cost / Error calculation, calculating the error margin
        error_margin = target_value - output_neurons[0]

        if (error_margin[0] < 0):
            error_margin[0] = error_margin[0] * -1
        
        #if the error margin is greater than the error threshold it is a bad fact
        if(error_margin > error_threshold):
            #add bad fact to counter
            bad_facts = bad_facts + 1

            #calculating the output delta
            delta_output = delta_function_output(output_neurons, target_value)

            #calculating the change in weights
            change_in_weights = learn_rate * (delta_output @ np.transpose(hidden_neurons))

            #storing the old weight of the hidden neuron
            weights_old = weights_hidden_to_output

            #calculating the new hidden to output weights
            weights_hidden_to_output = change_in_weights + weights_old

            #calculating the new bias to output weights
            bias_hidden_to_output += learn_rate * delta_output

            #calculating the hidden delta
            delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden_neurons * (1 - hidden_neurons))

            #calculating the new input to hidden weights
            change_in_weights = learn_rate * (delta_hidden @ np.transpose(input_values))

            #old weights
            weights_old = weights_input_to_hidden

            #the new weights
            weights_input_to_hidden = change_in_weights + weights_old

            #calculating the new bias to hidden weights
            bias_input_to_hidden += learn_rate * delta_hidden
        
        else:
            #add good fact to counter
            good_facts = good_facts + 1

    #add the epoch iteration number to the array
    epoch_array.append(epoch)

    #add the epoch iteration bad facts number to the array
    bad_facts_array.append(bad_facts)

    # Show results for this epoch
    print("Epoch: " + str(epoch) + ", Good facts:" + str(good_facts) + ", Bad facts: " + str(bad_facts))

#print the graph of bad facts vs epochs

# plotting the points
plt.plot(epoch_array, bad_facts_array)
  
# naming the x axis
plt.xlabel('Epochs')

# naming the y axis
plt.ylabel('Bad facts')
  
# giving a title to my graph
plt.title('Bad Facts vs Epochs')
  
# function to show the graph
plt.show()

# saving the input to hidden weights to a file
np.savez("weights_ith_normal.npz", weights_input_to_hidden)

# saving the array weights to a file
np.savez("weights_hto_normal.npz", weights_hidden_to_output)

# loading the input to hidden weights
npzfile = np.load("weights_ito_normal.npz")
weights_input_to_hidden = npzfile["arr_0"]

#checking the correctness of the weights with testing data

#importing the testing data
testing_dataset = pandas.read_csv('testing_dataset.csv')

#the target output for the training set
target_output = testing_dataset.Survived.values

#we need to correct the fare column, some numbers have two points in the column
fare_array = testing_dataset.Fare.values

#iterating through each value in the fare column
for x in fare_array:
    dot_counter = 0
    original_value = x

    for character in x:
        if character =='.':
            dot_counter = dot_counter + 1

    if (dot_counter>1):
        #in the case there are more than 1 '.'
        x = x.replace('.','',1)

        testing_dataset['Fare'] = testing_dataset['Fare'].replace({original_value: x})

#the survived column is not part of the input data, therefore we remove it
testing_data_without_survived = testing_dataset.drop(['Survived'], axis=1)

#normalise the data
scaler = preprocessing.MinMaxScaler()
names = testing_data_without_survived.columns
d = scaler.fit_transform(testing_data_without_survived)
normalised_testing_data = pandas.DataFrame(d, columns=names)
normalised_testing_data.head()

#convert the testing data into a matrix with the use of numpy
normalised_matrix = np.asarray(normalised_testing_data)

#feed forward process
#good facts counter
good_facts = 0

#bad facts counter
bad_facts = 0

print("--- Training dataset results: ---")

#iterating through the records of the dataset
for record in range(len(normalised_matrix)):
    input_values = normalised_matrix[record]

    #changing the array into a vertical matrix of arrays which are size 1
    input_values = np.reshape(input_values, (len(input_values), 1))

    #getting the target value for the current record
    target_value = target_output[record]

    # multiply the weights with the input data, afterwards add the bias weights, 
    # the result is before the summation 4 arrays for each hidden neuron

    #multiply weights with input values
    product_input = np.dot(weights_input_to_hidden, input_values)

    hidden_neurons_pre = bias_input_to_hidden + (product_input)

    # calculate the values of the hidden neurons using the activation function
    hidden_neurons = sigmoid_function(hidden_neurons_pre) #hidden neuron values

    # multiply the weights with the hidden neurons, afterwards add the bias weights, 
    # the result is before the summation, 1 arrays for each hidden neuron

    #multiply weights with input values
    product_hidden = np.dot(weights_hidden_to_output, hidden_neurons)

    output_neurons_pre = bias_hidden_to_output + (product_hidden) # multiply the weights with the hidden neurons, afterwards add the bias weights

    #calculate the values of the output neurons using the activation function
    output_neurons = sigmoid_function(output_neurons_pre)

    # Cost / Error calculation, calculating the error margin
    error_margin = target_value - output_neurons[0]

    if (error_margin[0] < 0):
        error_margin[0] = error_margin[0] * -1
    
    #if the error margin is greater than the error threshold it is a bad fact
    if(error_margin > 0.5):
        #add bad fact to counter
        bad_facts = bad_facts + 1
    
    else:
        #add good fact to counter
        good_facts = good_facts + 1

    print("Good facts:" + str(good_facts) + ", Bad facts: " + str(bad_facts))

