from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt



def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    
    return results

def to_natural(label,n_bits):

    for i in range(len(label)):
        label_bin = bin(label[i])
        print(label_bin)

def create_nn(output_activation,output_neurons):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_neurons, activation=output_activation))
    
    return model

def main():

    #Get the Reuters dataset
    #There is 8982 training examples and 2246 test examples
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
        num_words=10000)

    #In this code, we use three kinds of loss functions. The first one is the
    #baseline used in the Section 3.5 from "Deep Learning with Python", Francois
    #Chollet. The second and third are alternative loss functions to compare the
    #accuracy.  
    loss_functions = ['categorical_crossentropy', 'mean_squared_error', 
                      'binary_crossentropy']
    output_neurons = [46, 46, 6]
    output_activation = ['softmax', 'softmax', 'sigmoid']

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    #Update the labels to follow the one hot encoding
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels) 
    
    to_natural(train_labels,6)
    #Create a dataset for validation
    # x_val = x_train[:1000]
    # partial_x_train = x_train[1000:]
    # y_val = one_hot_train_labels[:1000]
    # partial_y_train = one_hot_train_labels[1000:]

    # for k in range(2):
    #     model = create_nn(output_activation[k], output_neurons[k])
    #     model.summary()
        # model.compile(optimizer='rmsprop', loss=loss_functions[k],
        #             metrics=['accuracy'])
        # history = model.fit(partial_x_train, partial_y_train, epochs=9, 
        #                     batch_size=512, validation_data=(x_val, y_val))
        # results[i] = model.evaluate(x_test, one_hot_test_labels)
    
    # print(results)
    
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # plt.clf()
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

main()
        