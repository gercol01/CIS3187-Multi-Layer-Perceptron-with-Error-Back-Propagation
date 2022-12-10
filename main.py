from decimal import Decimal
from sklearn import preprocessing
import pandas #used to import the training data
import numpy as np

#Neural network must have 5 input neurons, 4 hidden neurons, and 1 output neuron
#Use the Sigmoid transfer function, 0.2 error threshold and 0.2 learning rate
#Split the dataset into training data and testing data by a factor of 80% to 20%

#our transfer function is sigmoid
def sigmoid_function(x):
    return (1/(1 + np.exp(-x)))

def logistic_deriv(x):
    return sigmoid_function(x) * (1-sigmoid_function(x))

#our error threshold is 0.2
error_threshold = 0.2

# our learning rate is 0.2
learning_rate = 0.2

#five neurons for the input layer
input_layer = 5

#four neurons for the output layer
hidden_layer = 4

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

#------ this is done the first time to intiate the training and testing data ------

# #reading all the dataset
# dataset = pandas.read_csv('titanic_dataset.csv')

# #randomly selecting 80% of the dataset
# dataset = dataset.sample(frac = 0.8)

# #writing the randomly selected data to a different csv file
# dataset.to_csv("training_dataset.csv", index=False)

#importing the training data
training_data = pandas.read_csv('training_dataset.csv')

#we need to correct the fare column, some numbers have two points in the column
fare_array = training_data.Fare.values

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

        training_data['Fare'] = training_data['Fare'].replace({original_value: x})

#normalise the data
scaler = preprocessing.MinMaxScaler()
names = training_data.columns
d = scaler.fit_transform(training_data)
scaled_df = pandas.DataFrame(d, columns=names)
scaled_df.head()

#the target output for the training set
target_output = training_data.Survived.values

#the survived column is not part of the input data, therefore we remove it
training_data = training_data.drop(['Survived'], axis=1)

#convert the training data into a matrix with the use of numpy
training_data = np.asarray(training_data)

training_count = len(training_data[:,0])

#initially set bad facts to -1
bad_facts = -1

#feed forward processing algorithm
while bad_facts != 0:
    #at the start of each epoch, reset the bad facts counter
    bad_facts = 0

    #at the start of each epoch, reset the good facts counter
    good_facts = 0

    counter_i = 0

    #iterate through the different records of training data
    for i in training_data:

        #get the input_values
        input_values = training_data[i]

        #empty array that will store the values of the hidden neurons
        neurons_hidden = np.zeros(hidden_layer)

        #empty array that will store the values of the output neurons
        neurons_output = np.zeros(1)

        counter_j = 0

        #we multiply the input neurons with the weights (input to hidden) in order to achieve the hidden neurons
        for j in hidden_layer:
            sum = 0

            counter_k = 0

            #calculate the value of the new hidden neuron
            for k in input_values:
                sum = sum + (k*weights_input_to_hidden[counter_k])
                counter_k = counter_k + 1

            #add the hidden neuron value
            neurons_hidden[counter_j] = sigmoid_function(sum)

            counter_j = counter_j + 1


        counter_j = 0

        #we multiply the hidden neurons with the weights (hidden to output) in order to achieve the output neurons
        for j in hidden_layer:
            sum = 0

            counter_k = 0

            #calculate the value of the new output neuron
            for k in neurons_hidden:
                sum = sum + (k*weights_hidden_to_output[counter_k])
                counter_k = counter_k + 1

            #add the hidden neuron value
            neurons_output[counter_j] = sigmoid_function(sum)

            counter_j = counter_j + 1

        #check the output if it is a good fact or a bad fact
        error_margin = target_output[counter_i] - neurons_output[0]

        #if negative make positive
        if(error_margin<0):
            error_margin * -1

        #if the error is greater than the error threshold, it is a bad fact
        if(error_margin>error_threshold):
            bad_facts = bad_facts + 1

            #error back propagation algorithm
        else:
            good_facts = good_facts + 1
        

        

        counter_i = counter_i + 1




            







#save the weights


#printing the column headers
#print (training_data.keys())





