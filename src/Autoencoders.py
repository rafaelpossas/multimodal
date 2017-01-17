from keras.layers import Input, Dense, Activation
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

def ex_1():
    # this is the size of our encoded representations

    input_img = Input(shape=(784,))
    encoded_1 = Dense(100, activation='relu')(input_img)

    decoded = Dense(784, activation='relu')(encoded_1)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)

    encoder_1 = Model(input=input_img, output=encoded_1)

    # create a placeholder for an encoded (32-dimensional) input
    #encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)
    print("Training the model...")
    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=2)
    print("Training Complete")
    # encode and decode some digits
    # note that we take them from the *test* set
    #encoded_imgs = encoder.predict(x_test)
    #decoded_imgs = decoder.predict(encoded_imgs)

    # n = 10  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()
    my_w = autoencoder.layers[1].get_weights()
    n = my_w[0].shape[1]  # how many digits we will display
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(my_w[0].T[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def ex_2():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    nb_epoch=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))


def ex_3():
    hidden_size = 196
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    unlabeled_train_index = np.argwhere(y_train >= 5)
    labeled_train_index = np.argwhere(y_train < 5)

    labeled_test_index = np.argwhere(y_test < 5)

    unlabeled_data = x_train[unlabeled_train_index, :]
    #unlabeled_data = unlabeled_data.astype('float32') / 255
    unlabeled_data = unlabeled_data.reshape(len(unlabeled_data), np.prod(unlabeled_data.shape[1:]))

    labeled_data_x = x_train[labeled_train_index, :]
    #labeled_data_x = labeled_data_x.astype('float32') / 255
    labeled_data_y = to_categorical(y_train[labeled_train_index])

    labeled_data_x = labeled_data_x.reshape(len(labeled_data_x), np.prod(labeled_data_x.shape[1:]))

    x_test = x_test[labeled_test_index, :]
    #x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

    y_test = y_test[labeled_test_index]
    y_test = to_categorical(y_test)

    print('# examples in unlabeled set: {0:d}\n'.format(unlabeled_data.shape[0]))
    print('# examples in supervised training set: {0:d}\n'.format(labeled_data_x.shape[0]))

    input_img = Input(shape=(784,))
    encoded = Dense(hidden_size, activation='relu',
                    activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print("Training on Unlabeled data...")
    autoencoder.fit(unlabeled_data, unlabeled_data,
                    nb_epoch=50,
                    batch_size=1000,
                    shuffle=True,
                    verbose=2)

    train_features = encoder.predict(labeled_data_x)
    test_features = encoder.predict(x_test)

    softmax = Sequential()
    softmax.add(Dense(input_dim=train_features.shape[1], output_dim=5, activation='sigmoid'))
    softmax.add(Dense(output_dim=5, activation='softmax'))
    softmax.compile(loss='categorical_crossentropy', optimizer='adadelta')

    print("Training Softmax on labeled data...")
    softmax.fit(train_features, labeled_data_y,
                nb_epoch=100,
                batch_size=1000,
                shuffle=True,
                verbose=2)

    scores = softmax.evaluate(test_features, y_test, verbose=0)
    print("Accuracy: {}".format(scores*100))



ex_3()