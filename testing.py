from sklearn import preprocessing
import pandas #used to import training data from an Excel file

#let us now import the training data 
training_data = pandas.read_csv('titanic_dataset.csv')

#we need to correct the fare column, some numbers have two points in the column
fare_array = training_data.Fare.values

#traverse all the values in the column
special_characters = "."

# x = "hello"

# x = x. replace("l", "-", 2)
# x = x. replace("-", "l", 1)
# x = x. replace("-", "", 1)

array = []

for x in fare_array:
    dot_counter = 0
    original_value = x

    for character in x:
        if character ==".":
            dot_counter = dot_counter + 1

    if (dot_counter>1):
        #in the case there are more than 1 '.'
        x = x.replace(".","",1)

        training_data['Fare'] = training_data['Fare'].replace({original_value: x})

    # # writing into the file
    # training_data.to_csv("titanic_dataset.csv", index=False)

    

# training_data.loc['Fare'] = array

scaler = preprocessing.MinMaxScaler()
names = training_data.columns
d = scaler.fit_transform(training_data)
scaled_df = pandas.DataFrame(d, columns=names)
scaled_df.head()

print ("hello")