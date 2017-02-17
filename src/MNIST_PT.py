from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras import models
from keras.layers import Dense
from keras.models import Sequential

np.random.seed(1337) # for reproducibility

def train_SAE(X_train, X_test, hidden_layers, do_pretrain=True,
    hidden_activation='tanh', output_activation='linear',
    batch_size=256, pre_train_epoch=10, nb_epoch=25):

    #activation = 'tanh'
    # Layer-wise pre-training
    pre_trained_layers = []
    X_train_tmp = X_train
    for n_in, n_out in zip(hidden_layers[:-1], hidden_layers[1:]):
        print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))

        # Create AE and training
        encoder = Sequential([Dense(n_out, input_dim=n_in,
                          activation=hidden_activation)])
        decoder = Sequential([Dense(n_in, input_dim=n_out,
                          activation=hidden_activation)])

        autoencoder = Sequential()
        autoencoder.add(encoder)
        autoencoder.add(decoder)
        autoencoder.output_reconstruction = True

        # Train the simple autoencoder
        ae_model = Sequential()
        ae_model.add(autoencoder)
        ae_model.compile(loss='mse', optimizer='rmsprop')
        ae_model.fit(X_train_tmp, X_train_tmp, batch_size=batch_size,
                          nb_epoch=pre_train_epoch)

        # Store trainined weight
        pre_trained_layers.append((ae_model.layers[0].layers[0],
                          ae_model.layers[0].layers[0].get_weights()))

        # Predict hidden nodes output
        autoencoder.output_reconstruction = False
        ae_model.compile(loss='mse', optimizer='rmsprop')

        # Update training data
        X_train_tmp = ae_model.predict(X_train_tmp)

    # Fine-tuning
    print('Fine-tuning')
    # Create a new model
    model = models.Sequential()
    # Add the pre-trained layers to the model
    for layer, weights in pre_trained_layers:
        model.add(layer)
        model.layers[-1].set_weights(weights)

    # Add the output layer with linear activation function
    model.add(Dense(hidden_layers[0], input_dim=hidden_layers[-1],
        activation=output_activation))

    # Train the model
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, validation_data=(X_test, X_test))

    score = model.evaluate(X_test, X_test, verbose=1)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    return score

if __name__ == '__main__':

    batch_size = 640
    nb_classes = 10
    nb_epoch = 3
    hidden_layers = [784, 600]

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    score = train_SAE(X_train, X_test, hidden_layers, do_pretrain=True,
                        hidden_activation='tanh', output_activation='linear',
                        batch_size=256, pre_train_epoch=2, nb_epoch=2)