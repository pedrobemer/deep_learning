from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def case_1(loss, val_loss, acc, val_acc, epochs):

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (One Hot Entropy)')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def case_2(loss, val_loss, acc, val_acc, epochs):

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def case_3(loss, val_loss, acc, val_acc, epochs):

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Entropy)')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy for the Baseline')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_graph(case,loss, val_loss, acc, val_acc, epoch):

    epochs = range(1, epoch + 1)
    switch_case = {
        1: case_1,
        2: case_2,
        3: case_3,
    }

    func = switch_case.get(case, "There isn't this case")
    func(loss[case-1], val_loss[case-1], acc[case-1], val_acc[case-1], epochs)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    
    return results

def to_natural(label, n_bits):
    natural_label = np.zeros((len(label), n_bits))

    for i in range(len(label)):
        label_bin = bin(label[i])
        label_bin = ['0'] * (n_bits - len(label_bin[2:])) + list(label_bin[2:])
        for k in range(len(label_bin)):
            natural_label[i][k] = label_bin[k]

    return natural_label


def create_nn(output_activation, output_neurons):
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
    epoch = 9

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    #Update the labels to follow the one hot encoding
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    #Update the labels to follow the binary natural representation
    #(0-46 in decimals)
    natural_train_labels = to_natural(train_labels, 6)
    natural_test_labels = to_natural(test_labels, 6)
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val_one_hot = one_hot_train_labels[:1000]
    partial_y_train_one_hot = one_hot_train_labels[1000:]
    y_val_natural = natural_train_labels[:1000]
    partial_y_train_natural = natural_train_labels[1000:]

    results = np.zeros((3,2))
    loss = np.zeros((3,epoch))
    val_loss = np.zeros((3,epoch))
    acc = np.zeros((3,epoch))
    val_acc = np.zeros((3,epoch))
    for k in range(len(output_neurons)):
        model = create_nn(output_activation[k], output_neurons[k])
        model.summary()
        model.compile(optimizer='rmsprop', loss=loss_functions[k],
                    metrics=['accuracy'])
        if (k < 2):
            history = model.fit(partial_x_train, partial_y_train_one_hot,
                                epochs=epoch, batch_size=512,
                                validation_data=(x_val, y_val_one_hot))
            results[k] = model.evaluate(x_test, one_hot_test_labels)
        else:
            history = model.fit(partial_x_train, partial_y_train_natural,
                                epochs=9, batch_size=512,
                                validation_data=(x_val, y_val_natural))
            results[k] = model.evaluate(x_test, natural_test_labels)
        loss[k] = history.history['loss']
        val_loss[k] = history.history['val_loss']
        acc[k] = history.history['accuracy']
        val_acc[k] = history.history['val_accuracy']

    plot_graph(2,loss, val_loss, acc, val_acc, epoch)
    print(results)
    


main()
        