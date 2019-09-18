# To split the dataset into train and test datasets
from math import cos,pi
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt
import csv

def create_nn(num_hidden_layers, neurons):
    activation = 'relu'
    model = Sequential()
    print("Hidden Layers: ", num_hidden_layers, "Neurons per Hidden Layer: ", neurons)
    if (num_hidden_layers > 3):
        return -1

    for i in range(num_hidden_layers):
        if (i == 0):
            model.add(Dense(neurons, input_dim=1, activation=activation))
            if (num_hidden_layers >= 2):
                model.add(BatchNormalization())
        else:
            model.add(Dense(neurons, activation=activation))
            if (num_hidden_layers == 3 and i != 2):
                model.add(BatchNormalization())
    model.add(Dense(1, activation="linear"))
    return model
                

def main():
    hidden_layers = [1,2,3]

    with open('mhasker_paper_results.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(['MSE', 'Total Parameters', 'Total Units', 'Hidden Layers'])
        for h in hidden_layers:
            #Equation
            y = lambda x : 2*((2*(cos(x)**2)-1)**2)

            #Uniform samples for x in the above equation
            total_samples = int(120e3)
            x_samples = np.random.uniform(-2*pi,2*pi,total_samples)
            x_features_train = x_samples[0:int(total_samples/2)]
            y_label_train = list(map(y, x_features_train))
            x_features_test = x_samples[int(total_samples/2):int(total_samples)]
            y_label_test = list(map(y, x_features_test))

            if (h == 1):
                neurons = [24,48,72,128,256]
            elif (h == 2):
                neurons = [12,24,36]
            elif (h == 3):
                neurons = [8,16,24]
            for j in range(len(neurons)):
                model = create_nn(h,neurons[j])
                print(model.count_params())
                if (model == -1):
                    print("Configuration Error")
                else:
                    model.summary()
                    sgd = optimizers.SGD(lr=0.0001, momentum=0.9)
                    model.compile(loss='mean_squared_error', optimizer=sgd)
                    model.fit(x_features_train, y_label_train, epochs=2000, batch_size=3000)
                    loss = model.evaluate(x_features_test, y_label_test)
                    spamwriter.writerow([loss, model.count_params(), h*neurons[j]],h)


main()
#Neural Network Configuration
# hidden_layers_activation = "relu"
# one_hidden_layer = [24,48,72,128,256]
# two_hidden_layers = [12,24,36]
# three_hidden_layers = [8,16,24]

# model = Sequential()

