
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

topology = [len(X_train[0]), 200, 100, len(Y_train[0])]
batch_size = 128
nb_pretraining_epochs = 00
nb_epoch = 2



autoencoder_0 = Sequential()

encoder_0 = Sequential([Dense(200, input_dim=784)])
decoder_0 = Sequential([Dense(784, input_dim=200)])

autoencoder_0.add(encoder_0)
autoencoder_0.add(decoder_0)
autoencoder_0.output_reconstruction = True

autoencoder_0_model = Sequential()
autoencoder_0_model.add(autoencoder_0)
autoencoder_0_model.compile(optimizer='rmsprop', loss='mse')
autoencoder_0_model.fit(X_train, X_train, batch_size=batch_size, nb_epoch = nb_pretraining_epochs)

temp_0 = Sequential()
temp_0.add(encoder_0)
temp_0.compile(loss='mse', optimizer=Adam(lr=0.001))

X_train_1 = temp_0.predict(X_train)

autoencoder_1 = Sequential()
encoder_1 = Sequential([Dense(input_dim = 200, output_dim=100)])
decoder_1 = Sequential([Dense(input_dim = 100, output_dim=200)])

autoencoder_1.add(encoder_1)
autoencoder_1.add(decoder_1)
autoencoder_1.output_reconstruction = True

autoencoder_1_model = Sequential()
autoencoder_1_model.add(autoencoder_1)
autoencoder_1.compile(optimizer ="rmsprop", loss = 'mse')
autoencoder_1.fit(X_train_1, X_train_1, batch_size = batch_size, nb_epoch =  nb_pretraining_epochs)

model = Sequential()
model.add(encoder_0)
model.add(encoder_1)
model.add(Dense(input_dim = 100, output_dim = 10, init = 'zero', activation = 'softmax'))


adam = Adam(lr = 0.001)
model.compile(optimizer = adam, loss = "categorical_crossentropy", metrics=['accuracy'])

model.fit(x = X_train,
          y = Y_train,
          batch_size = batch_size,
          nb_epoch = nb_epoch,
          verbose = 2,
          validation_data = (X_test, Y_test),
          shuffle = True)