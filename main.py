from decimal import Decimal
from sklearn import preprocessing
import pandas #used to import the training data
import numpy as np
import matplotlib.pyplot as plt

#Neural network must have 5 input neurons, 4 hidden neurons, and 1 output neuron
#Use the Sigmoid transfer function, 0.2 error threshold and 0.2 learning rate
#Split the dataset into training data and testing data by a factor of 80% to 20%

#our transfer/ activation function is sigmoid
def sigmoid_function(x):
    return (1/(1 + np.exp(-x)))

#derivative of sigmoid function
def derivative_sigmoid_function(x):
    return sigmoid_function(x)*(1 - sigmoid_function(x))

#delta function used during error back propagation for output neuron
def delta_function_output(output, target):
    return (output) * (1 - output) * (target - output)

#delta function used during error back propagation for hidden neuron
def delta_function_hidden(output, weight, delta):
    return (output) * (1- output) * (weight) * (delta)

#function to change weights during error back propagation
def weights_function(learning_rate, delta, output):
    return (learning_rate) * (delta) * (output)

def logistic_deriv(x):
    return sigmoid_function(x) * (1-sigmoid_function(x))

#epoch array
epoch_array = []

#bad facts array
bad_facts_array = []

#our error threshold is 0.2
error_threshold = 0.2

# our learning rate is 0.2
learning_rate = 0.2

#five neurons for the input layer
input_layer = 5

#four neurons for the hidden layer
hidden_layer = 4

#one neuron for the output layer
output_layer =1

#counter for the epochs, i.e. how many iterations of the training date there is
epoch_count = 1

np.random.seed(1) #this statement causes the random values to be the same every time you run the program.

#instantiating the weights of the input to hidden layer, weights are instantiated to a number between -1 & 1

#each hidden layer neuron has an ammount of weights equal to the number of input neurons, in this case 5 weights for each of the 4 hidden neurons
weights_input_to_hidden = np.random.uniform(-1, 1, (input_layer,hidden_layer))

#each output layer neuron has an amount of weights equal to the number of hidden neurons, in this case 4 weights for the 1 output neuron
weights_hidden_to_output = np.random.uniform(-1, 1, (hidden_layer, 1))

#empty array for pre activation
pre_activation_array = np.zeros(hidden_layer)

#empty array for post activation
post_activation_array = np.zeros(hidden_layer)

# #------ this is done the first time to intiate the training and testing data ------

# #reading all the dataset
# dataset = pandas.read_csv('titanic_dataset.csv')

# #randomly select 80% of a list of numbers equal to the length of the dataset
# random_selection = np.random.rand(len(dataset)) <= 0.8

# #training data is 80%
# training_dataset = dataset[random_selection]

# #testing data is the inverse - 20%
# testing_dataset = dataset[~random_selection]

# #instantiating the training dataset csv
# training_dataset.to_csv("training_dataset.csv", index=False)

# #instantiating the testing dataset csv
# testing_dataset.to_csv("testing_dataset.csv", index=False)

#importing the training data
training_dataset = pandas.read_csv('training_dataset.csv')

#importing the testing data
testing_dataset = pandas.read_csv('testing_dataset.csv')

#the target output for the training set
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

#normalise the data
scaler = preprocessing.MinMaxScaler()
names = training_data_without_survived.columns
d = scaler.fit_transform(training_data_without_survived)
normalised_training_data = pandas.DataFrame(d, columns=names)
normalised_training_data.head()

#convert the training data into a matrix with the use of numpy
normalised_matrix = np.asarray(normalised_training_data)

training_count = len(normalised_matrix[:,0])

#initially set bad facts to -1, i.e. the number of incorrect outcomes
bad_facts = -1

#epoch counter
epoch_counter = 0

#feed forward processing algorithm
while bad_facts != 0:
    #increment epoch counter
    epoch_counter = epoch_counter + 1

    #at the start of each epoch, reset the bad facts counter
    bad_facts = 0

    #at the start of each epoch, reset the good facts counter
    good_facts = 0

    #numerical counter for loop
    counter_i = 0

    #iterate through the different records of training data
    for i in normalised_matrix:

        #target variable
        target = target_output[counter_i]

        #get the input_values
        input_values = normalised_matrix[counter_i]

        #empty array that will store the values of the hidden neurons, in this case 4 neurons
        neurons_hidden = np.zeros(hidden_layer)

        #empty array that will store the values of the output neurons, in this case 1 output neuron
        neurons_output = np.zeros(output_layer)

        #numerical counter used to calculate the hidden neurons
        counter_j = 0

        #we multiply the input neurons with the weights (input to hidden) in order to calculate the values of the hidden neurons
        for j in range(hidden_layer):
            #sum of the weights multiplied by the input neurons before the sigmoid function is called
            sum = 0

            #numerical counter for loop
            counter_k = 0

            #calculate the value of the new hidden neuron
            for k in input_values:
                #the input value
                input_value = k

                #getting the weights of that input value
                weight_array = weights_input_to_hidden[counter_k]

                #retreiving the weight corresponding to that hidden neuron
                weight = weight_array[counter_j]

                #hidden neuron sum before activation, is input neuron value multiplied by weight
                sum = sum + (input_value * weight)

                #counter to select next input value
                counter_k = counter_k + 1

            #instantiate the hidden neuron value by using the sigmoid function on the sum
            neurons_hidden[counter_j] = sigmoid_function(sum)

            #counter to calculate the next hidden neuron
            counter_j = counter_j + 1


        #numerical counter used to calculate the output neuron
        counter_j = 0

        #we multiply the hidden neurons with the weights (hidden to output) in order to achieve the output neurons
        for j in range(output_layer):
            #sum of the weights multiplied by the hidden neurons
            sum = 0

            #numerical counter used to select hidden neurons
            counter_k = 0

            #calculate the value of the new output neuron
            for k in neurons_hidden:
                #value for hidden neuron
                hidden_neuron = k

                #getting the weight that corresponds to the output neuron
                weight = weights_hidden_to_output[counter_k]

                #calculating the sum before activation function
                sum = sum + (hidden_neuron * weight) #this was hidden_neuron by k 
                counter_k = counter_k + 1

            #add the outut neuron value
            neurons_output[counter_j] = sigmoid_function(sum)

            #counter used to select the next output neuron
            counter_j = counter_j + 1

        #check the output if it is a good fact or a bad fact by calculating the error margin
        error_margin = target - neurons_output[0]

        #if negative make positive
        if(error_margin < 0):
            error_margin * -1

        #if the error is greater than the error threshold, it is a bad fact
        if(error_margin > error_threshold):
            bad_facts = bad_facts + 1

            #error back propagation algorithm

            #array to store the deltas for the output layer neurons
            delta_output = np.zeros(output_layer)

            #array to store the deltas for the hidden layer neurons
            delta_hidden = np.zeros(hidden_layer)

            #numerical counter used to point to the output neurons
            counter_o = 0
            
            #calculate delta 'δ' for output
            for o in neurons_output:
                #calculating the delta of the output neuron
                delta_output[counter_o] = delta_function_output(neurons_output[0], target)

                #counter used to point to the next output neuron
                counter_o = counter_o + 1

            #counter used to point to the hidden to output weights
            counter_weights_hto = 0

            #starting with the weights hidden to output
            for weights in weights_hidden_to_output:
                #selecting the delta value for the output neuron
                delta_value = delta_output[0]

                #calculating the change in weights
                change_in_weights = weights_function(learning_rate, delta_value, neurons_hidden[counter_weights_hto])

                #storing the old weight of the hidden neuron
                weight_old = weights_hidden_to_output[counter_weights_hto]

                #calculating the new weight
                weight_new = change_in_weights + weight_old

                #changing the weight 
                weights_hidden_to_output[counter_weights_hto] = weight_new

                #counter used to point to next hidden to output weight
                counter_weights_hto = counter_weights_hto + 1


            #numerical counter used to point to the hidden neurons
            counter_h = 0
            
            #calculate delta 'δ' for hidden neurons
            for h in neurons_hidden:
                #neuron value
                neuron_value = h

                #calculating the delta of the hidden neuron
                delta_hidden[counter_h] = delta_function_hidden(neuron_value, weights_hidden_to_output[counter_h], delta_output[0])

                #counter used to point to the next hidden neuron
                counter_h = counter_h + 1
            
            #counter used to point to weights
            counter_weights_ith = 0

            #changing the weights input to hidden, for each delta there is, we must change the weights pointing to it
            for delta in delta_hidden:
                #getting the delta value
                delta_value = delta_hidden[counter_weights_ith]

                #counter used to points to the input neurons which have weights pointing to the hidden neuron
                counter_neuron_input = 0

                for neuron in range(input_layer):
                    change_in_weights = weights_function(learning_rate, delta_value, input_values[counter_neuron_input])

                    weights_old_array = weights_input_to_hidden[counter_neuron_input]

                    weight_old = weights_old_array[counter_weights_ith]

                    weight_new = change_in_weights + weight_old

                    #updating the weights with the new weight
                    weights_input_to_hidden[counter_neuron_input][counter_weights_ith] = weight_new

                    counter_neuron_input = counter_neuron_input + 1
                
                counter_weights_ith = counter_weights_ith + 1


        else:
            good_facts = good_facts + 1

            #if it is a good fact, there is no need for error back propagation
        
        #next set of values from the training data
        counter_i = counter_i + 1

    #add epoch number to epoch array
    epoch_array.append(epoch_counter)

    #add bad facts number to bad facts array
    bad_facts_array.append(bad_facts)

    #after an epoch the bad facts and good facts are output        
    print("Epoch: " + str(epoch_counter) + ". There are " + str(bad_facts) + " bad facts, " + str(good_facts) + " good facts.")

#print the graph of bad facts vs epochs

# plotting the points
plt.plot(epoch_array, bad_facts_array)
  
# naming the x axis
plt.xlabel('epochs')

# naming the y axis
plt.ylabel('bad facts')
  
# giving a title to my graph
plt.title('Bad Facts vs Epochs')
  
# function to show the plot
plt.show()
    

        




            







#save the weights


#printing the column headers
#print (training_data.keys())





